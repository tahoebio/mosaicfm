import datasets
from pathlib import Path
from tqdm.autonotebook import tqdm
from scgpt.tokenizer import GeneVocab

from datasets.utils.logging import disable_progress_bar
from datasets import disable_caching
disable_progress_bar()
disable_caching()

VERSION = "2023-12-15"
DATASET_NAME = f"cellxgene_primary_{VERSION}"
chunk_id = 0
PATH = f"/vevo/cellxgene/"
vocab = GeneVocab.from_file("/vevo/cellxgene/cellxgene_primary_2023-12-15_vocab.json")
special_tokens = ["<pad>", "<cls>", "<eoc>"]
for s in special_tokens:
    if s not in vocab:
        vocab.append_token(s)
CLS_TOKEN = vocab["<cls>"]
PAD_VALUE = -2
def append_cls(example):
    example["genes"].insert(0, CLS_TOKEN)
    example["expressions"].insert(0, PAD_VALUE)
    return example

def get_files(path, dataset_name):
    files = [str(f.resolve()) for f in Path(path).glob(f"{dataset_name}*.dataset")]
    files = [f for f in files if "_cls_appended" not in f]
    return files

def get_chunks(path, dataset_name):
    files = get_files(path, dataset_name)
    for file in files:
        dataset = datasets.load_from_disk(file)
        yield dataset

for file_path in tqdm(get_files(PATH,DATASET_NAME)):
    new_file_path = file_path.split(".dataset")[0] + "_cls_appended.dataset"
    if Path(new_file_path).exists():
        continue
    small_dataset = datasets.load_from_disk(file_path)
    cls_appended = small_dataset.map(append_cls, num_proc=8)
    cls_appended.save_to_disk(new_file_path)