# Copyright (C) Vevo Therapeutics 2025. All rights reserved.
# python
import gc
import logging
import os
from typing import Any, Dict, Generator, List, Optional

import datasets
import numpy as np
import pandas as pd
import scanpy as sc
from omegaconf import DictConfig
from omegaconf import OmegaConf as om
from scipy.sparse import csc_matrix, csr_matrix
from mosaicfm.data import CountDataset
from mosaicfm.tokenizer import GeneVocab


def find_h5ad_files(
    directory: str,
    ignore_subdirs: Optional[List[str]] = None,
) -> List[str]:
    """Recursively search for all '.h5ad' files in a given directory while
    ignoring specified subdirectories.

    Args:
        directory (str): The root directory to search for .h5ad files.
        ignore_subdirs (Optional[List[str]]): A list of subdirectory names to exclude from the search.
            Defaults to an empty list if not provided.

    Returns:
        List[str]: A list of full paths to files with the '.h5ad' extension.
    """
    h5ad_files = []
    ignore_subdirs = ignore_subdirs or []
    for root, dirs, files in os.walk(directory):
        # Filter out directories that should be ignored
        dirs[:] = [d for d in dirs if d not in ignore_subdirs]
        for file in files:
            if file.endswith(".h5ad") or file.endswith(".h5ad.gz"):
                h5ad_files.append(os.path.join(root, file))
    return h5ad_files


def dataset_generator(
    adata_files: List[str],
    vocab: GeneVocab,
    gene_col: str,
    cls_token: str,
    pad_value: int,
    obs_filter: Optional[Dict[str, str]] = None,
    obs_metadata_columns: Optional[List[str]] = None,
    additional_metadata_info: Optional[List[Dict[str, Any]]] = None,
    add_obs_index: bool = True,
) -> Generator[Dict, None, None]:
    """Generator that processes AnnData files and yields dictionary items for
    the dataset.

    The function reads each AnnData file, applies optional filters, merges additional
    metadata if provided, and constructs a CountDataset for each file. It can optionally
    add the dataframe index as an observation metadata column.

    Args:
        adata_files (List[str]): List of AnnData file paths.
        vocab (GeneVocab): A GeneVocab object used to encode gene IDs.
        gene_col (str): Column name within observation variables corresponding to genes.
        cls_token (str): Special token representing the class token.
        pad_value (int): Value used for padding in the dataset.
        obs_filter (Optional[Dict[str, str]]): Dictionary for filtering observations.
        obs_metadata_columns (Optional[List[str]]): Columns to include in observation metadata.
        additional_metadata_info (Optional[List[Dict[str, Any]]]):
            List of dictionaries containing paths and merge keys for additional metadata.
        add_obs_index (bool): If True, resets the index of the observations and adds it
            as a metadata key.

    Yields:
        Dict: A dictionary representing an item in the dataset, including count data and metadata.
    """
    if obs_metadata_columns is None:
        obs_metadata_columns = []

    for file in adata_files:
        adata = sc.read_h5ad(file, backed="r")

        if obs_filter:
            filter_key = obs_filter.get("key")
            filter_value = obs_filter.get("value")
            if filter_key and filter_value:
                adata = adata[adata.obs[filter_key] == filter_value]

        base_obs = adata.obs.copy()
        if add_obs_index:
            index_key = (
                base_obs.index.name if base_obs.index.name is not None else "index"
            )
            base_obs.reset_index(inplace=True)
            if index_key not in obs_metadata_columns:
                obs_metadata_columns.append(index_key)

        if additional_metadata_info:
            for meta_source in additional_metadata_info:
                metadata_df = pd.read_csv(meta_source["path"])
                left_key = meta_source["merge_keys"]["adata_key"]
                right_key = meta_source["merge_keys"]["metadata_key"]
                base_obs.loc[:, left_key] = (
                    base_obs.loc[:, left_key].astype(str).str.strip()
                )
                metadata_df.loc[:, right_key] = (
                    metadata_df.loc[:, right_key].astype(str).str.strip()
                )
                base_obs = base_obs.merge(
                    metadata_df[[right_key, *meta_source["columns"]]],
                    left_on=left_key,
                    right_on=right_key,
                    how="left",
                )
                for col in meta_source["columns"]:
                    if col not in obs_metadata_columns:
                        obs_metadata_columns.append(col)
        if obs_metadata_columns:
            base_obs.loc[:, obs_metadata_columns] = (
                base_obs.loc[:, obs_metadata_columns].fillna("").astype(str)
            )
        base_var = adata.var.copy()
        base_var.reset_index(inplace=True)
        gene_ids_in_vocab = np.array([vocab[gene] for gene in base_var[gene_col]])
        count_matrix = adata.X
        if isinstance(count_matrix, np.ndarray):
            count_matrix = csr_matrix(count_matrix)
        elif isinstance(count_matrix, csc_matrix):
            count_matrix = count_matrix.tocsr()
        elif hasattr(count_matrix, "to_memory"):
            count_matrix = count_matrix.to_memory().tocsr()
        else:
            raise ValueError(f"count_matrix must be a numpy ndarray or csr_matrix, found {type(count_matrix)}")

        count_dataset = CountDataset(
            count_matrix,
            gene_ids_in_vocab,
            cls_token_id=vocab[cls_token],
            pad_value=pad_value,
        )

        for idx, item in enumerate(count_dataset):
            final_metadata = {
                col: base_obs.iloc[idx][col] for col in obs_metadata_columns
            }
            item.update(final_metadata)
            yield item

        del adata, count_matrix
        gc.collect()


def main(cfg: DictConfig) -> None:
    log = logging.getLogger(__name__)
    logging.basicConfig(
        format=r"%(asctime)s: [%(process)d][%(threadName)s]: %(levelname)s: %(name)s: %(message)s",
        level=logging.INFO,
    )
    adata_files = find_h5ad_files(cfg.huggingface.adata_dir, cfg.huggingface.ignore_dir)
    if len(adata_files) == 0:
        raise ValueError("Found no .h5ad files in the specified directory.")
    vocab_file = os.path.join(
        cfg.huggingface.vocab_output_root,
        cfg.huggingface.vocab_path,
    )
    vocab = GeneVocab.from_file(vocab_file)
    gene_col = cfg.huggingface.gene_col
    num_chunks = cfg.huggingface.get("num_chunks", 10)
    chunks = np.array_split(adata_files, num_chunks)
    obs_filter = cfg.huggingface.get("obs_filter", None)
    obs_metadata_columns = cfg.huggingface.get("obs_metadata_columns", None)
    additional_metadata_info = cfg.huggingface.get("additional_metadata_info", None)
    add_obs_index = cfg.huggingface.get("add_obs_index", True)

    for i, chunk in enumerate(chunks):
        save_path = os.path.join(cfg.huggingface.output_root, f"chunk_{i}.dataset")
        if os.path.exists(save_path):
            log.info(f"Chunk {i} dataset already exists. Skipping.")
            continue
        log.info(f"Processing chunk {i} with {len(chunk)} files")
        chunk_dataset = datasets.Dataset.from_generator(
            dataset_generator,
            gen_kwargs={
                "adata_files": chunk.tolist(),
                "vocab": vocab,
                "gene_col": gene_col,
                "cls_token": cfg.huggingface.get("cls_token", "<cls>"),
                "pad_value": cfg.huggingface.get("pad_value", -2),
                "obs_filter": obs_filter,
                "obs_metadata_columns": obs_metadata_columns,
                "additional_metadata_info": additional_metadata_info,
                "add_obs_index": add_obs_index,
            },
            num_proc=min(len(chunk), cfg.huggingface.get("num_proc", 1)),
            keep_in_memory=False,
        )
        chunk_dataset.save_to_disk(
            save_path,
            num_proc=cfg.huggingface.get("num_proc", 1),
        )
        log.info(f"Chunk {i} dataset saved to disk with length: {len(chunk_dataset)}")
        chunk_dataset.cleanup_cache_files()
        del chunk_dataset
        gc.collect()
    log.info("Script execution completed.")


if __name__ == "__main__":
    import sys

    yaml_path = sys.argv[1]
    with open(yaml_path) as f:
        cfg = om.load(f)
    om.resolve(cfg)
    main(cfg)
