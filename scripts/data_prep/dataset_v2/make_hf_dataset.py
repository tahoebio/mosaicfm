# Copyright (C) Vevo Therapeutics 2024-2025. All rights reserved.
import logging
import os
from typing import Dict, List, Optional

import datasets
import numpy as np
import scanpy as sc
from omegaconf import DictConfig
from omegaconf import OmegaConf as om

from mosaicfm.data import CountDataset
from mosaicfm.tokenizer import GeneVocab


def find_h5ad_files(directory: str, ignore_subdirs: Optional[List] = None) -> List[str]:
    h5_files = []
    if ignore_subdirs is None:
        ignore_subdirs = []
    # Walk through all directories and files in the provided directory
    for root, dirs, files in os.walk(directory):
        dirs[:] = [d for d in dirs if d not in ignore_subdirs]
        for file in files:
            # Check if the file ends with .h5
            if file.endswith(".h5ad"):
                # Append the full path of the file to the list
                h5_files.append(os.path.join(root, file))
    return h5_files


def dataset_generator(
    adata_files: List[str],
    vocab: GeneVocab,
    gene_col: str,
    cls_token: str,
    pad_value: int,
) -> Dict:
    for chunk_id, file in enumerate(adata_files):
        adata = sc.read_h5ad(file)
        gene_ids_in_vocab = np.array([vocab[gene] for gene in adata.var[gene_col]])
        count_matrix = (
            adata.X.toarray() if not isinstance(adata.X, np.ndarray) else adata.X
        )
        torch_dataset = CountDataset(
            count_matrix,
            gene_ids_in_vocab,
            cls_token_id=vocab[cls_token],
            pad_value=pad_value,
        )
        for item in torch_dataset:
            yield item


def main(cfg: DictConfig) -> None:
    # Configure environment and logging

    # Logging setup
    log = logging.getLogger(__name__)
    logging.basicConfig(
        format="%(asctime)s: [%(process)d][%(threadName)s]: %(levelname)s: %(name)s: %(message)s",
        level=logging.INFO,
    )
    adata_files = find_h5ad_files(cfg.huggingface.adata_dir, cfg.huggingface.ignore_dir)
    vocab = GeneVocab.from_file(
        os.path.join(cfg.huggingface.vocab_output_root, cfg.huggingface.vocab_path),
    )
    gene_col = cfg.huggingface.gene_col
    chunks = np.array_split(adata_files, 10)
    for i, chunk in enumerate(chunks):
        save_path = os.path.join(cfg.huggingface.output_root, f"chunk_{i}.dataset")
        if os.path.exists(save_path):
            log.info(f"Chunk {i} dataset already exists. Skipping.")
            continue
        log.info(f"Processing chunk {i} with  {len(chunk)} files")
        chunk_dataset = datasets.Dataset.from_generator(
            dataset_generator,
            gen_kwargs={
                "adata_files": chunk.tolist(),
                "vocab": vocab,
                "gene_col": gene_col,
                "cls_token": cfg.huggingface.get("cls_token", "<cls>"),
                "pad_value": cfg.huggingface.get("pad_value", -2),
            },
            num_proc=len(chunk),
            keep_in_memory=True,
        )
        chunk_dataset.save_to_disk(save_path, num_proc=8)
        log.info(f"Chunk {i} dataset saved to disk with length: {len(chunk_dataset)}")
        chunk_dataset.cleanup_cache_files()
        del chunk_dataset
    log.info("Script execution completed.")


if __name__ == "__main__":
    import sys

    yaml_path = sys.argv[1]
    with open(yaml_path) as f:
        cfg = om.load(f)
    om.resolve(cfg)
    main(cfg)
