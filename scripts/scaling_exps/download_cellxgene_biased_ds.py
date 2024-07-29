import json
import argparse
import os

import cellxgene_census
import logging
import pandas as pd
import scanpy as sc
from tqdm.autonotebook import tqdm


def chunker(seq, size):
    return (seq[pos : pos + size] for pos in range(0, len(seq), size))

def process_cellxgene_data(version, old_vocab_path, dest_dir):

    log = logging.getLogger(__name__)
    logging.basicConfig(
        # Example of format string
        # 2022-06-29 11:22:26,152: [822018][MainThread]: INFO: Message here
        format=f"%(asctime)s: [%(process)d][%(threadName)s]: %(levelname)s: %(name)s: %(message)s"
    )
    logging.getLogger(__name__).setLevel("INFO")  # Train script

    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir, exist_ok=True)

    with cellxgene_census.open_soma(census_version=version) as census:
        cell_metadata = census["census_data"]["homo_sapiens"].obs.read(
            column_names=["is_primary_data", "soma_joinid", "suspension_type", "dataset_id"]
        )
        gene_metadata = census["census_data"]["homo_sapiens"].ms["RNA"].var.read()
        gene_metadata = gene_metadata.concat().to_pandas()
        # Concatenates results to pyarrow.Table
        cell_metadata = cell_metadata.concat()

        # # Converts to pandas.DataFrame
        cell_metadata = cell_metadata.to_pandas()

        # filter out the cells from Madisoon study
        #239222 cells satisfy this coming from spleen, Esophagus Epithelium, and Lung Parenchyma tissues
        cell_metadata = cell_metadata[((cell_metadata['dataset_id']=='644a578d-ffdc-446b-9679-e7ab4c919c13')
                | (cell_metadata['dataset_id']=='b07fb54c-d7ad-4995-8bb0-8f3d8611cabe') 
                | (cell_metadata['dataset_id']=='2672b679-8048-4f5e-9786-f1b196ccfd08'))]
        
    with open(old_vocab_path, "r") as f:
        old_gene_dict = json.load(f)
    gene_to_id_table =  pd.read_csv("https://github.com/bowang-lab/scGPT/files/13243634/gene_info.csv")
    id_to_gene_old = {
        gene_id: gene_name
        for gene_name, gene_id in zip(
            gene_to_id_table["feature_name"], gene_to_id_table["feature_id"]
        )
    }

    new_gene_ids = gene_metadata["feature_id"].to_list()
    id_to_gene_new = {
        gene_id: gene_name
        for gene_name, gene_id in zip(
            gene_metadata["feature_name"], gene_metadata["feature_id"]
        )
    }

    new_gene_list = [
        id_to_gene_old.get(gene_id, id_to_gene_new[gene_id]) for gene_id in new_gene_ids
    ]
    master_gene_to_id = {
        gene_name: gene_id for gene_name, gene_id in zip(new_gene_list, new_gene_ids)
    }
    master_id_to_gene = {
        gene_id: gene_name for gene_name, gene_id in zip(new_gene_list, new_gene_ids)
    }
    expanded_dict = old_gene_dict.copy()
    starting_num = max(old_gene_dict.values()) + 1
    for new_gene in new_gene_list:
        if new_gene not in old_gene_dict.keys():
            expanded_dict[new_gene] = starting_num
            starting_num += 1
    vocab_path = os.path.join(dest_dir, f"cellxgene_primary_{version}_vocab.json")
    with open(vocab_path, "w") as f:
        json.dump(expanded_dict, f, indent=2)

    gene_to_id_path = os.path.join(dest_dir, f"gene_info_{version}.json")

    with open(gene_to_id_path, "w") as f:
        json.dump(master_gene_to_id, f, indent=2)

    obs_coords = cell_metadata[
        (cell_metadata["is_primary_data"] == True)
        & (cell_metadata["suspension_type"] != "na")
    ]["soma_joinid"].tolist()
    log.info(f"Number of unique cells in {version} data: {len(obs_coords)}")

    N = 10000
    chunk_size = 200000
    dataset_size = len(obs_coords)
    main_table_key = "counts"
    token_col = "feature_name"


    with cellxgene_census.open_soma(census_version=version) as census:
        for chunk_id, chunk_indices in tqdm(
            enumerate(chunker(obs_coords, chunk_size)), total=dataset_size // chunk_size + 1
        ):
            save_path = os.path.join(dest_dir, f"chunk_{chunk_id}.h5ad")
            adata = cellxgene_census.get_anndata(
                census,
                organism="Homo sapiens",
                obs_coords=chunk_indices,
            )
            sc.pp.filter_genes(adata, min_counts=(3 / 10000) * N)
            updated_feature_name = [
                master_id_to_gene[gene_id] for gene_id in adata.var["feature_id"]
            ]
            adata.var["feature_name"] = updated_feature_name
            adata.write_h5ad(save_path)
            log.info(f"Chunk {chunk_id} saved")




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process Cellxgene data.")
    parser.add_argument("--version", required=True, help="Desired version of the cellxgene.")
    parser.add_argument("--old_vocab", required=True, help="Path to the previous vocab file")
    parser.add_argument("--dest_dir", required=True, help="The destination directory to save the output.")

    args = parser.parse_args()

    process_cellxgene_data(args.version, args.old_vocab, args.dest_dir)