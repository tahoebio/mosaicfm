# Copyright (C) Vevo Therapeutics 2024. All rights reserved.
import json
import logging
import sys

import numpy as np
import pandas as pd
import scanpy as sc
from datasets import Dataset
from omegaconf import DictConfig, OmegaConf

from mosaicfm.tokenizer import GeneVocab

# Logging setup
log = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s: [%(process)d][%(threadName)s]: %(levelname)s: %(name)s: %(message)s",
)
logging.getLogger(__name__).setLevel("INFO")


def generate_gene_aliases(gene_info_path: str) -> pd.DataFrame:
    log.info(f"Generating gene aliases from {gene_info_path}...")

    raw_df = pd.read_csv(gene_info_path, sep="\t")
    synonym_to_symbol = {
        synonym: row["Symbol"]
        for row in raw_df.to_dict(orient="records")
        for synonym in row["Synonyms"].split("|")
    }

    # Ensure all symbols are included as their own synonym
    synonym_to_symbol.update(
        {
            value: value
            for value in synonym_to_symbol.values()
            if value not in synonym_to_symbol
        },
    )

    # Convert dictionary to DataFrame
    df = pd.DataFrame(
        list(synonym_to_symbol.items()),
        columns=["synonym", "gene_symbol"],
    )
    df.set_index("synonym", inplace=True)

    log.info(f"Generated gene alias DataFrame with {len(df)} entries.")
    return df


def record_generator(
    adata: sc.AnnData,
    perturbation_metadata: pd.DataFrame,
    cfg: DictConfig,
):
    ctrl_adata = adata[adata.obs["perturbation"] == "ctrl"]
    perturbation_list = list(set(adata.obs["perturbation"]))
    log.info(f"Using {len(perturbation_list)} perturbations")

    for perturbation_name in perturbation_list:
        perturb_adata = adata[adata.obs["perturbation"] == perturbation_name]
        log.info(f"Retrieved {len(perturb_adata)} cells for {perturbation_name}")
        perturbation_edist = perturbation_metadata.loc[perturbation_name, "edist"]
        depmap_dependency = perturbation_metadata.loc[
            perturbation_name,
            "depmap_dependency",
        ]
        perturbation_targets = [
            perturbation_metadata.loc[perturbation_name, "target_gene_vocab_id"],
        ]

        for cell in perturb_adata:
            expressions_perturbed = cell.X.A[0]
            genes_pert = cell.var["id_in_vocab"].values

            random_ids = np.random.randint(
                low=0,
                high=len(ctrl_adata),
                size=cfg.num_ctrl_samples_to_pair,
            )

            assert all(target in genes_pert for target in perturbation_targets)

            for ctrl_id in random_ids:
                expressions_ctrl = ctrl_adata[ctrl_id].X.A[0]

                yield {
                    "depmap_dependency": np.float32(depmap_dependency),
                    "perturbation_edist": np.float32(perturbation_edist),
                    "perturbation_target_genes": np.array(
                        perturbation_targets,
                        dtype=np.int32,
                    ),
                    "expressions_ctrl_raw": np.array(
                        expressions_ctrl,
                        dtype=np.float32,
                    ),
                    "expressions_perturbed_raw": np.array(
                        expressions_perturbed,
                        dtype=np.float32,
                    ),
                    "genes": np.array(genes_pert, dtype=np.int32),
                    "cell_line": "K562",
                    "perturbation_name": perturbation_name,
                    "cell_key": cell.obs.index.values[0],
                    "cell_key_ctrl": ctrl_adata[ctrl_id].obs.index[0],
                }


def main(cfg: DictConfig) -> None:
    log.info("Starting main script execution...")

    # Load vocab and gene mapping
    log.info(f"Loading vocab from {cfg.vocab_path}...")
    vocab = GeneVocab.from_file(cfg.vocab_path)
    log.info(f"Vocab loaded with size {len(vocab)}")

    log.info(f"Loading gene-to-ensembl mapping from {cfg.gene_to_ensembl_path}...")
    with open(cfg.gene_to_ensembl_path) as f:
        gene_to_ensembl = json.load(f)
    log.info(f"Gene-to-ensembl mapping loaded with {len(gene_to_ensembl)} entries")

    ensembl_to_gene_name = {v: k for k, v in gene_to_ensembl.items()}

    # Generate gene aliases
    gene_alias_df = generate_gene_aliases(cfg.gene_info_path)

    # Load in raw data
    log.info(f"Loading raw data from {cfg.raw_data_path}...")
    adata = sc.read_h5ad(cfg.raw_data_path)
    adata.var["gene_name"] = adata.var.index  # Add gene_name column to var
    log.info(f"Raw data loaded. Data shape: {adata.shape}")

    # Load metadata
    log.info(f"Loading Adamson metadata from {cfg.adamson_metadata_path}...")
    metadata_df = pd.read_csv(cfg.adamson_metadata_path)
    metadata_df = metadata_df[
        ~metadata_df["perturbation"].isin(["Gal4-4(mod)", "63(mod)"])
    ]  # Filter out these perturbations if present, these don't map to a gene
    log.info(f"Adamson metadata loaded with {len(metadata_df)} records.")

    metadata_df["target_gene_name"] = metadata_df["perturbation"].apply(
        lambda x: x.split("+")[0].strip(),
    )

    log.info("Mapping target genes to vocab IDs...")
    mapped_id = []
    for gene_name in metadata_df["target_gene_name"]:
        if gene_name in vocab:
            mapped_id.append(vocab[gene_name])
        elif gene_name in gene_alias_df.index:
            mapped_id.append(vocab[gene_alias_df.loc[gene_name, "gene_symbol"]])
        else:
            mapped_id.append(vocab["<pad>"])
    metadata_df["target_gene_vocab_id"] = mapped_id
    assert (
        vocab["<pad>"]
        not in metadata_df[metadata_df["perturbation"] != "ctrl"][
            "target_gene_vocab_id"
        ]
    ), "Some target genes were not found in the vocabulary."

    metadata_df = metadata_df.set_index("perturbation", drop=False)

    # Load DepMap dependency score
    log.info(f"Loading DepMap dependency scores from {cfg.depmap_scores_path}...")
    depmap_df = pd.read_csv(cfg.depmap_scores_path)
    depmap_df = depmap_df.rename(columns={"Unnamed: 0": "cell_line_depmap_id"})

    k562_depmap_id = cfg.k562_depmap_id
    k562_dependency_scores = (
        depmap_df.set_index("cell_line_depmap_id")
        .loc[k562_depmap_id, :]
        .transpose()
        .to_dict()
    )  # Extract K562 dependency scores as dictionary with gene symbol as key
    k562_dependency_scores = {
        k.split(" (")[0]: v for k, v in k562_dependency_scores.items()
    }

    adamson_dep_scores = []
    for gene_symbol in metadata_df["target_gene_name"]:
        if gene_symbol in k562_dependency_scores:
            adamson_dep_scores.append(k562_dependency_scores[gene_symbol])
        elif gene_symbol in gene_alias_df.index:
            gene_alias = gene_alias_df.loc[gene_symbol, "gene_symbol"]
            adamson_dep_scores.append(k562_dependency_scores.get(gene_alias, None))
        else:
            adamson_dep_scores.append(None)
    metadata_df["depmap_dependency"] = adamson_dep_scores
    log.info("DepMap dependency scores added to perturbation metadata.")

    # Data preparation
    log.info("Starting data preparation...")
    adata.var["id_in_vocab"] = [
        vocab[ensembl_to_gene_name.get(gene_id, "<pad>")]
        for gene_id in adata.var["ensembl_id"]
    ]
    gene_ids_in_vocab = np.array(adata.var["id_in_vocab"])
    filter_vocab = adata.var["id_in_vocab"] != vocab["<pad>"]
    log.info(
        f"Matched {np.sum(filter_vocab)} / {len(gene_ids_in_vocab)} genes in vocabulary of size {len(vocab)}",
    )
    adata = adata[:, filter_vocab]

    log.info("Identifying highly variable genes...")
    sc.pp.highly_variable_genes(
        adata,
        n_top_genes=cfg.n_top_genes,
        subset=False,
        flavor="seurat_v3",
    )
    filter_hvg = (
        adata.var["gene_name"].isin(metadata_df["target_gene_name"])
        | adata.var["highly_variable"]
    )
    log.info(
        f"Subset to {np.sum(filter_hvg)} / {len(adata.var)} after HVG selection and adding back target genes",
    )
    adata = adata[:, filter_hvg]

    # Create Dataset using from_generator
    log.info("Creating Hugging Face dataset using from_generator...")
    hf_dataset = Dataset.from_generator(
        lambda: record_generator(
            adata=adata,
            perturbation_metadata=metadata_df,
            cfg=cfg,
        ),
    )
    hf_dataset.set_format(type="torch")
    log.info(f"Generated Adamson dataset with {len(hf_dataset)} records.")

    dataset_save_path = cfg.dataset_save_path
    log.info(f"Saving Adamson dataset to {dataset_save_path}...")
    hf_dataset.save_to_disk(
        dataset_save_path,
        max_shard_size=cfg.get("max_shard_size", "200MB"),
    )
    log.info(f"Saved Adamson dataset to {dataset_save_path}")


if __name__ == "__main__":
    yaml_path = sys.argv[1]

    # Disable resolving environment variables through omegaconf.
    OmegaConf.clear_resolver("oc.env")

    # Load yaml file.
    log.info(f"Loading configuration from {yaml_path}...")
    with open(yaml_path) as f:
        cfg = OmegaConf.load(f)
    OmegaConf.resolve(cfg)
    main(cfg)
    log.info("Script execution completed.")
