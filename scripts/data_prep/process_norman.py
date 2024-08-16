# Copyright (C) Vevo Therapeutics 2024. All rights reserved.
import json
import logging
import sys

import numpy as np
import pandas as pd
import scanpy as sc
from datasets import Dataset
from omegaconf import DictConfig, OmegaConf
from tqdm.auto import tqdm

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
    df.loc["KIAA1804"] = "MAP3K21"  # Add missing gene alias

    log.info(f"Generated gene alias DataFrame with {len(df)} entries.")
    return df


def map_gene_names_to_vocab(gene_name_list, vocab, gene_alias_df):
    vocab_map_per_row = []
    for gene_name in gene_name_list:
        if gene_name in vocab:
            vocab_map_per_row.append(vocab[gene_name])
        elif gene_name in gene_alias_df.index:
            vocab_map_per_row.append(vocab[gene_alias_df.loc[gene_name, "gene_symbol"]])
        else:
            return None
    return vocab_map_per_row if vocab_map_per_row else None


def map_gene_name_to_dep(gene_name_list, dep_scores, gene_alias_df):
    dep_scores_per_row = []
    for gene_name in gene_name_list:
        if gene_name in dep_scores:
            dep_scores_per_row.append(dep_scores[gene_name])
        elif gene_name in gene_alias_df.index:
            gene_alias = gene_alias_df.loc[gene_name, "gene_symbol"]
            dep_scores_per_row.append(dep_scores[gene_alias])
        else:
            return None
    return dep_scores_per_row if dep_scores_per_row else None


def record_generator(
    adata: sc.AnnData,
    perturbation_metadata: pd.DataFrame,
    cfg: DictConfig,
):
    ctrl_adata = adata[adata.obs["perturbation"] == "ctrl"]
    perturbation_list = list(set(adata.obs["perturbation"]))
    log.info(f"Using {len(perturbation_list)} perturbations")

    for perturbation_name in tqdm(perturbation_list):
        perturb_adata = adata[adata.obs["perturbation"] == perturbation_name]
        log.info(f"Retrieved {len(perturb_adata)} cells for {perturbation_name}")
        perturbation_edist = perturbation_metadata.loc[perturbation_name, "edist"]
        depmap_dependency = perturbation_metadata.loc[
            perturbation_name,
            "depmap_dependency",
        ]
        perturbation_targets = perturbation_metadata.loc[
            perturbation_name,
            "target_gene_vocab_id",
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
                genes_ctrl = ctrl_adata[ctrl_id].var["id_in_vocab"].values
                assert all(genes_pert == genes_ctrl)

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
    adata.var["gene_name"] = adata.var.index
    adata.var = adata.var.rename(columns={"ensemble_id": "ensembl_id"})
    log.info(f"Raw data loaded. Data shape: {adata.shape}")

    # Load metadata
    log.info(f"Loading Norman metadata from {cfg.norman_metadata_path}...")
    norman_df = pd.read_csv(cfg.norman_metadata_path)
    norman_df["target_gene_names"] = norman_df["perturbation"].apply(
        lambda x: [
            gene_name.strip() for gene_name in x.split("+") if gene_name != "ctrl"
        ],
    )
    norman_df["target_gene_vocab_id"] = norman_df["target_gene_names"].apply(
        lambda gene_name_list: map_gene_names_to_vocab(
            gene_name_list,
            vocab,
            gene_alias_df,
        ),
    )
    norman_df = norman_df.set_index("perturbation", drop=False)
    assert not any(
        norman_df[norman_df["perturbation"] != "ctrl"]["target_gene_vocab_id"].isna(),
    )
    log.info(f"Norman metadata loaded with {len(norman_df)} records.")

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
    )
    k562_dependency_scores = {
        k.split(" (")[0]: v for k, v in k562_dependency_scores.items()
    }

    norman_df["depmap_dependency"] = norman_df["target_gene_names"].apply(
        lambda gene_name_list: map_gene_name_to_dep(
            gene_name_list,
            k562_dependency_scores,
            gene_alias_df,
        ),
    )
    log.info("DepMap dependency scores added to Norman metadata.")

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
    target_genesymbol_set = [
        gene for gene_list in norman_df["target_gene_names"] for gene in gene_list
    ]
    filter_hvg = (
        adata.var["gene_name"].isin(target_genesymbol_set)
        | adata.var["highly_variable"]
    )
    log.info(
        f"Subset to {np.sum(filter_hvg)} / {len(adata.var)} after HVG selection and adding back target genes",
    )
    adata = adata[:, filter_hvg]

    # Create Dataset using from_generator
    log.info("Creating Hugging Face dataset using from_generator...")
    mosaic_dataset = Dataset.from_generator(
        lambda: record_generator(
            adata=adata,
            perturbation_metadata=norman_df,
            cfg=cfg,
        ),
    )
    mosaic_dataset.set_format(type="torch")
    log.info(f"Generated mosaic dataset with {len(mosaic_dataset)} records.")

    dataset_save_path = cfg.dataset_save_path
    log.info(f"Saving mosaic dataset to {dataset_save_path}...")
    mosaic_dataset.save_to_disk(
        dataset_save_path,
        max_shard_size=cfg.get("max_shard_size", "200MB"),
    )
    log.info(f"Saved mosaic dataset to {dataset_save_path}")


if __name__ == "__main__":
    yaml_path = sys.argv[1]

    OmegaConf.clear_resolver("oc.env")
    log.info(f"Loading configuration from {yaml_path}...")
    with open(yaml_path) as f:
        cfg = OmegaConf.load(f)
    OmegaConf.resolve(cfg)
    main(cfg)
    log.info("Script execution completed.")
