import datasets
from streaming import MDSWriter
from streaming.base.util import merge_index
from pathlib import Path
from typing import Iterable, Tuple
from multiprocessing import Pool
import os


# Initialize the worker process
def init_worker():
    # Get the pid for the current worker process
    pid = os.getpid()
    print(f'\nInitialize Worker PID: {pid}', flush=True, end='')


def get_files(path: str, dataset_name: str) -> Iterable[str]:
    files = [str(f.resolve()) for f in Path(path).glob(f"{dataset_name}*_cls_appended.dataset")]
    return files


def each_task(out_root: str, dataset_root_path: str, dataset_name: str) -> Tuple[str, str]:
    for dataset_path in get_files(dataset_root_path, dataset_name):
        chunk_suffix = dataset_path.rstrip("_cls_appended.dataset").split("_")[-1]
        sub_out_root = f"{out_root}/chunk_{chunk_suffix}"
        yield sub_out_root, dataset_path


def convert_to_mds(args: Iterable[Tuple[str, str]]) -> None:
    sub_out_root, dataset_path = args
    dataset = datasets.load_from_disk(dataset_path)
    with MDSWriter(out=sub_out_root, columns=columns,
                   compression=compression, hashes=hashes) as out:
        for sample in dataset:
            out.write(sample)


VERSION = "2023-12-15"
DATASET_NAME = f"cellxgene_primary_{VERSION}"
PATH = "/vevo/cellxgene/"
out_root = f"/vevo/cellxgene/streaming_dataset_all_chunks"
num_files = len(get_files(PATH, DATASET_NAME))
columns = {
    'genes': 'pkl',
    'id': 'int64',
    'expressions': 'pkl'
}
# Compression algorithm name
compression = 'zstd'

# Hash algorithm name
hashes = 'sha1', 'xxh64'

arg_tuples = each_task(out_root, PATH, DATASET_NAME)

# Process group of data in parallel into directories of shards.
with Pool(initializer=init_worker, processes=num_files) as pool:
    for count in pool.imap(convert_to_mds, arg_tuples):
        pass
print("Finished Writing MDS Files. Merging Index...")
merge_index(out_root, keep_local=True)
print("Merging Index Complete.")