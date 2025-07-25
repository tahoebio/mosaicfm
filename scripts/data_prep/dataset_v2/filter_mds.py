# Copyright (C) Vevo Therapeutics 2025. All rights reserved.
import os
from multiprocessing import get_context
from pathlib import Path

from streaming import MDSWriter, StreamingDataset
from streaming.base.util import merge_index

# Configuration
NUM_WORKERS = 4  # Adjust depending on CPU/memory
MIN_UMI = 2000
MIN_NNZ = 1000

# Paths
LOCAL_DATASET_ROOT = "/tahoe/mosaicfm/datasets/vevo_merged_jan_2025/filtered_subsets/source_MDS/tahoe_100m_MDS_v2/valid"
FILTERED_DATASET_ROOT = "/tahoe/mosaicfm/datasets/vevo_merged_jan_2025/filtered_subsets/tahoe_100m_MDS_v2/valid"

# MDS column schema
COLUMNS = {
    "expressions": "pkl",
    "genes": "pkl",
    "id": "int64",
    "drug": "str",
    "Cell_ID_Cellosaur": "str",
    "canonical_smiles": "str",
    "pubchem_cid": "str",
}
COMPRESSION = "zstd"
HASHES = ["sha1", "xxh64"]

# Ensure output directory exists
os.makedirs(FILTERED_DATASET_ROOT, exist_ok=True)


def filter_chunk(chunk_path: str) -> str:
    chunk_name = os.path.basename(chunk_path.rstrip("/"))
    out_path = os.path.join(FILTERED_DATASET_ROOT, chunk_name)
    os.makedirs(out_path, exist_ok=True)

    print(f"Processing {chunk_name} ...")

    try:
        dataset = StreamingDataset(
            local=chunk_path,
            remote=None,
            allow_unsafe_types=True,
            shuffle=False,
            batch_size=1,
        )
    except RuntimeError as e:
        print(f"⚠️ Skipping {chunk_name}: {e}")
        return out_path

    num_written = 0
    with MDSWriter(
        out=out_path,
        columns=COLUMNS,
        compression=COMPRESSION,
        hashes=HASHES,
    ) as writer:
        for sample in dataset:
            expr = sample["expressions"].flatten()[1:]
            if expr.sum().item() >= MIN_UMI and expr.shape[0] >= MIN_NNZ:
                writer.write(sample)
                num_written += 1
    print(f"Finished {chunk_name}: {num_written}/{len(dataset)} samples written")
    return out_path


def main():
    # Only include chunks that don't already have filtered/index.json
    chunk_paths = []
    for chunk_dir in Path(LOCAL_DATASET_ROOT).glob("chunk_*"):
        if not chunk_dir.is_dir():
            continue
        chunk_name = chunk_dir.name
        filtered_index_path = Path(FILTERED_DATASET_ROOT) / chunk_name / "index.json"
        if not filtered_index_path.exists():
            chunk_paths.append(str(chunk_dir))

    print(f"🔁 Retrying {len(chunk_paths)} failed/missing chunks...")

    ctx = get_context("forkserver")
    with ctx.Pool(processes=min(len(chunk_paths), NUM_WORKERS)) as pool:
        pool.map(filter_chunk, chunk_paths)

    merge_index(FILTERED_DATASET_ROOT, keep_local=True)
    print(f"🎉 Done filtering and merging into: {FILTERED_DATASET_ROOT}")


if __name__ == "__main__":
    main()
