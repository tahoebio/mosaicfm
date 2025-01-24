import os
import scanpy as sc
import numpy as np
import json
import logging
from omegaconf import DictConfig
from omegaconf import OmegaConf as om
from mosaicfm.tokenizer import GeneVocab

# Logging setup
logging.basicConfig(
    format="%(asctime)s: [%(process)d][%(threadName)s]: %(levelname)s: %(name)s: %(message)s",
    level=logging.INFO,
)
log = logging.getLogger(__name__)

def find_h5ad_files(directory: str) -> list:
    h5_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".h5ad"):
                h5_files.append(os.path.join(root, file))
    return h5_files

def load_json_file(file_path: str) -> dict:
    with open(file_path) as file:
        return json.load(file)


def process_datasets(files: list, vocab: dict, gene_to_id: dict, possible_keys: list, output_dir: str):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    total_cells_passed_qc = 0
    datasets_passed_qc = 0
    ignored_datasets = []

    for file_path in files:
        adata = sc.read_h5ad(file_path, backed='r')
        total_genes = len(adata.var_names)
        matched = False
        feature_IDs = None

        # First, try matching using possible keys
        for key in possible_keys:
            if key in adata.var.columns:
                ids = adata.var[key].values
                matches = sum(id in vocab for id in ids)
                percentage_matched = (matches / total_genes) * 100
                print(f"Checking {file_path} using key '{key}': found {matches} matches.")
                if percentage_matched > 60:
                    matched = True
                    feature_IDs = ids
                    break

        # If not sufficiently matched, try remapping using the index
        if not matched:
            remapped_ids = [gene_to_id.get(gene, None) for gene in adata.var_names]
            matches = sum(id in vocab for id in remapped_ids if id is not None)
            percentage_matched = (matches / total_genes) * 100
            if matches > 0 and percentage_matched > 60:
                matched = True
                feature_IDs = remapped_ids

        if matched:
            datasets_passed_qc += 1
            total_cells_passed_qc += adata.n_obs
            # Subset adata to include only matched genes and add feature_ID column
            mask = [id in vocab for id in feature_IDs]
            adata_subset = adata[:, mask].to_memory().copy()
            adata_subset.var['feature_id'] = np.array(feature_IDs)[mask]
            # Save the subset
            output_file = os.path.join(output_dir, os.path.basename(file_path))
            adata_subset.write_h5ad(output_file)
            print(f"Included {file_path}: {percentage_matched:.2f}% genes matched, saved to {output_file}.")
        else:
            ignored_datasets.append(file_path)
            print(f"No valid gene IDs found or matched for {file_path}.")

    # Print total cells and datasets that passed QC
    print(f"Total cells across all datasets that passed QC: {total_cells_passed_qc}")
    print(f"Number of datasets that passed QC: {datasets_passed_qc}")
    if ignored_datasets:
        print("Ignored datasets:")
        for dataset in ignored_datasets:
            print(dataset)

def main(cfg: DictConfig):
    log.info("Starting processing datasets.")
    adata_root = cfg.scperturb.adata_root
    vocab = GeneVocab.from_file(os.path.join(cfg.vocab.output_root, cfg.vocab.output_file))
    id_to_gene = load_json_file(os.path.join(cfg.vocab.output_root, cfg.vocab.id_to_gene_output_file))
    gene_to_id = dict(zip(id_to_gene.values(), id_to_gene.keys()))
    possible_keys = cfg.scperturb.possible_gene_id_keys
    output_dir = cfg.scperturb.output_dir
    files = find_h5ad_files(adata_root)
    process_datasets(files, vocab, gene_to_id, possible_keys, output_dir)
    log.info("Script completed successfully.")

if __name__ == "__main__":
    import sys

    yaml_path = sys.argv[1]
    log.info(f"Loading configuration from {yaml_path}...")
    with open(yaml_path) as f:
        cfg = om.load(f)
    om.resolve(cfg)
    main(cfg)
