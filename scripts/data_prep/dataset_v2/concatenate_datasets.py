# Copyright (C) Vevo Therapeutics 2025. All rights reserved.
import logging
import os
import sys
from pathlib import Path
from typing import List, Generator
from multiprocessing import Process

import datasets
from omegaconf import DictConfig
from omegaconf import OmegaConf as om

# Logging setup
logging.basicConfig(
    format="%(asctime)s: [%(process)d][%(threadName)s]: %(levelname)s: %(name)s: %(message)s",
    level=logging.INFO,
)
log = logging.getLogger(__name__)

# Disable caching for efficiency
datasets.disable_caching()


def get_files(path: str) -> List[str]:
    """Retrieve dataset chunk file paths."""
    files = sorted(str(f.resolve()) for f in Path(path).glob("chunk*.dataset"))
    log.info(f"Found {len(files)} dataset chunks.")
    return files


def get_datasets(files: List[str]) -> Generator[datasets.Dataset, None, None]:
    """Lazy load datasets using a generator to prevent memory overload."""
    for file in files:
        yield datasets.load_from_disk(file)


def save_dataset_parallel(dataset: datasets.Dataset, path: str):
    """Save dataset to disk in a separate process for speed."""
    process = Process(target=dataset.save_to_disk, args=(path,))
    process.start()
    return process


def main(cfg: DictConfig):
    dataset_root = cfg.huggingface.output_root
    dataset_name = cfg.huggingface.dataset_name
    save_dir = cfg.huggingface.merged_dataset_root
    test_size = cfg.huggingface.split_parameters.test_size

    log.info(f"Merging dataset chunks from {dataset_root}...")

    # Concatenate datasets efficiently using a generator
    merged_dataset = datasets.concatenate_datasets(list(get_datasets(get_files(dataset_root))))

    log.info(f"Total {dataset_name} size: {len(merged_dataset)} samples")

    # Index-based train-test split (faster and avoids duplication)
    split_idx = int(len(merged_dataset) * (1 - test_size))
    train_dataset = merged_dataset.select(range(split_idx))
    test_dataset = merged_dataset.select(range(split_idx, len(merged_dataset)))

    log.info(f"Train set: {len(train_dataset)} samples")
    log.info(f"Test set: {len(test_dataset)} samples")

    # Ensure the save directory exists
    os.makedirs(save_dir, exist_ok=True)
    log.info(f"Saving datasets to {save_dir}...")

    # Save train and test datasets in parallel
    train_process = save_dataset_parallel(train_dataset, os.path.join(save_dir, "train.dataset"))
    test_process = save_dataset_parallel(test_dataset, os.path.join(save_dir, "valid.dataset"))

    train_process.join()
    test_process.join()

    log.info("Dataset merging and saving completed successfully.")


if __name__ == "__main__":
    yaml_path = sys.argv[1]
    log.info(f"Loading configuration from {yaml_path}...")
    with open(yaml_path) as f:
        cfg = om.load(f)
    om.resolve(cfg)
    main(cfg)
