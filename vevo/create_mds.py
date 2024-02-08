import datasets
from streaming import MDSWriter
from pathlib import Path
from tqdm.autonotebook import tqdm

def get_files(path, dataset_name):
    files = [str(f.resolve()) for f in Path(path).glob(f"{dataset_name}*_cls_appended.dataset")]
    return files

def get_chunks(path, dataset_name):
    files = get_files(path, dataset_name)
    for file in files:
        dataset = datasets.load_from_disk(file)
        yield dataset

VERSION = "2023-12-15"
DATASET_NAME = f"cellxgene_primary_{VERSION}"
PATH = "/vevo/cellxgene/"
out_root = f"/vevo/cellxgene/streaming_dataset_all_chunks"
num_files = len(get_files(PATH,DATASET_NAME))
dataloader_batchsize=1
columns = {
    'genes': 'pkl',
    'id': 'int64',
    'expressions': 'pkl'
}
# Compression algorithm name
compression = 'zstd'

# Hash algorithm name
hashes = 'sha1', 'xxh64'

with MDSWriter(out=out_root, columns=columns,
               compression=compression, hashes=hashes) as out:
    for dataset in tqdm(get_chunks(PATH,DATASET_NAME), total=num_files):
        for sample in dataset:
            out.write(sample)