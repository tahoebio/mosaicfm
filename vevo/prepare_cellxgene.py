import cellxgene_census
import scanpy as sc
import json
import scgpt.scbank
import os
from tqdm.autonotebook import tqdm

# query_name = "blood"
VERSION = "2023-12-15"
DATASET_NAME = f"cellxgene_primary_{VERSION}"
with cellxgene_census.open_soma(census_version=VERSION) as census:
    cell_metadata = census["census_data"]["homo_sapiens"].obs.read(column_names=["is_primary_data", "soma_joinid"])
    gene_metadata = census["census_data"]["homo_sapiens"].ms["RNA"].var.read(column_names=["feature_name"])
    # Concatenates results to pyarrow.Table
    cell_metadata = cell_metadata.concat()

    # Converts to pandas.DataFrame
    cell_metadata = cell_metadata.to_pandas()

new_gene_list = gene_metadata.concat().to_pandas()["feature_name"].to_list()
with open("../../scgpt/tokenizer/default_cellxgene_vocab.json", "r") as f:
    old_gene_dict = json.load(f)

print("old gene list length:", len(old_gene_dict))
expanded_dict = old_gene_dict.copy()
starting_num = max(old_gene_dict.values()) + 1
for new_gene in new_gene_list:
    if new_gene not in old_gene_dict.keys():
        expanded_dict[new_gene] = starting_num
        starting_num += 1
print("new gene dict length:", len(expanded_dict))
dump_path = "/vevo/cellxgene/cellxgene_primary_2023-12-15_vocab.json"
with open(dump_path, "w") as f:
    json.dump(expanded_dict, f, indent=2)
gene_vocab = scgpt.tokenizer.GeneVocab.from_dict(expanded_dict)
obs_coords = cell_metadata[cell_metadata["is_primary_data"] == True]["soma_joinid"].tolist()

N = 10000
chunk_size = 200000
dataset_size = len(obs_coords)
main_table_key = "counts"
token_col = "feature_name"


def chunker(seq, size):
    return (seq[pos:pos + size] for pos in range(0, len(seq), size))


with cellxgene_census.open_soma(census_version=VERSION) as census:
    for chunk_id, chunk_indices in tqdm(enumerate(chunker(obs_coords, chunk_size)),
                                        total=dataset_size // chunk_size + 1):
        if os.path.exists(f"{DATASET_NAME}_{chunk_id}.scb"):
            continue
        adata = cellxgene_census.get_anndata(census,
                                             organism="Homo sapiens",
                                             obs_coords=chunk_indices,
                                             )
        sc.pp.filter_genes(adata, min_counts=(3 / 10000) * N)
        adata.layers[main_table_key] = adata.X.copy()

        db = scgpt.scbank.DataBank.from_anndata(
            adata,
            vocab=gene_vocab,
            to=f"{DATASET_NAME}_{chunk_id}.scb",
            main_table_key=main_table_key,
            token_col=token_col,
            immediate_save=False,
        )
        dataset = db.main_data.data
        dataset.save_to_disk(f"{DATASET_NAME}_{chunk_id}.dataset")
