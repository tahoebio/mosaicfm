"""

Given an scGPT model, this script will create and save the cell line embeddings,
mean gene embeddings, and contextual gene embeddings needed to run the 
DepMap benchmarks.

"""

# imports
import argparse
import logging
import os
import numpy as np
import pandas as pd
import anndata as ad
import scanpy as sc
import torch
import utils
from tqdm import tqdm
from omegaconf import OmegaConf as om
from scgpt.model import ComposerSCGPTModel
from scgpt.data import DataCollator, CountDataset
from scgpt.tasks import get_batch_embeddings
from scgpt.tokenizer import GeneVocab

# set up logging
log = logging.getLogger(__name__)
logging.basicConfig(format=f"%(asctime)s: [%(process)d][%(threadName)s]: %(levelname)s: %(name)s: %(message)s")
logging.getLogger(__name__).setLevel("INFO")

# process mean gene embeddings from NPZ to create AnnData for training
def process_mean_gene_embs(base_path, model_name):

    # load genes and scores for marginal essentiality task
    genes, scores = utils.get_marginal_genes_scores(base_path)

    # load embeddings
    embs_npz = np.load(os.path.join(base_path, f"gene-embs/npz/{model_name}-gene-embs.npz"))
    mean_embs_all = embs_npz["gene_embeddings"]
    gene_names = embs_npz["gene_names"]
    log.info("loaded gene embeddings for processing")

    # get mean embeddings for each gene
    mean_embs = np.zeros((len(genes), mean_embs_all.shape[1]))
    for i, g in enumerate(tqdm(genes)):
        mean_embs[i] = mean_embs_all[np.where(gene_names == g)[0][0]]

    # create AnnData
    mean_embs_ad = ad.AnnData(
        X=mean_embs,
        obs=pd.DataFrame({"gene": genes, "score": scores}),
        var=pd.DataFrame({"dim": np.arange(mean_embs.shape[1])})
    )

    # write to disk
    outpath = os.path.join(base_path, f"gene-embs/{'_'.join(model_name.split('-'))}-mean-lt5gt70-bin.h5ad")
    mean_embs_ad.write_h5ad(outpath)
    log.info(f"saved mean gene embedding AnnData to {outpath}")

# generate and cell line embeddings, mean gene embeddings, and contextual gene embeddings
def main(base_path, model_path, model_name):
    
    # create paths
    model_config_path = os.path.join(model_path, "model_config.yml")
    vocab_path = os.path.join(model_path, "vocab.json")
    collator_config_path = os.path.join(model_path, "collator_config.yml")
    model_file = os.path.join(model_path, "best-model.pt")

    # load configurations and vocabulary
    model_config = om.load(model_config_path)
    collator_config = om.load(collator_config_path)
    vocab = GeneVocab.from_file(vocab_path)
    vocab.set_default_index(vocab["<pad>"])

    # load model
    model = ComposerSCGPTModel(model_config=model_config, collator_config=collator_config)
    model.load_state_dict(torch.load(model_file)["state"]["model"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    log.info(f"model loaded from {model_file}")

    # extract context-free embeddings
    with torch.no_grad(), torch.cuda.amp.autocast(enabled=True, dtype=torch.bfloat16):
        
        # load gene IDs and set step size
        gene2idx = vocab.get_stoi()
        all_gene_ids = np.array([[id for id in gene2idx.values()]])
        chunk_size = 30000

        # initialize empty array to hole embeddings
        num_genes = all_gene_ids.shape[1]
        gene_embeddings_ctx_free = (np.ones((num_genes, model_config["d_model"])) * np.nan)

        # iterate over genes
        for i in range(0, num_genes, chunk_size):

            # extract chunk of gene IDs
            chunk_gene_ids = all_gene_ids[:, i : i + chunk_size]
            chunk_gene_ids_tensor = torch.tensor(chunk_gene_ids, dtype=torch.long).to(device)

            # pass through model
            token_embs = model.model.gene_encoder(chunk_gene_ids_tensor)
            flag_embs = model.model.flag_encoder(
                torch.tensor(1, device=token_embs.device)
            ).expand(chunk_gene_ids_tensor.shape[0], chunk_gene_ids_tensor.shape[1], -1)
            total_embs = token_embs + flag_embs
            chunk_embeddings = model.model.transformer_encoder(total_embs)

            # bring to CPU and assign correctly
            chunk_embeddings_cpu = chunk_embeddings.to("cpu").to(torch.float32).numpy()
            gene_embeddings_ctx_free[i : i + chunk_size] = chunk_embeddings_cpu

    # cleanup
    torch.cuda.empty_cache()
    log.info("extracted context-free embeddings")

    # load and process AnnData of CCLE counts
    input_path = os.path.join(base_path, "counts.h5ad")
    adata = sc.read_h5ad(input_path)
    log.info(f"loaded CCLE AnnData from {input_path} for mean gene embeddings")
    adata.var["id_in_vocab"] = [vocab[gene] if gene in vocab else -1 for gene in adata.var["feature_name"]]
    adata = adata[:, adata.var["id_in_vocab"] >= 0]
    gene_ids_in_vocab = np.array(adata.var["id_in_vocab"])
    log.info(f"matched {np.sum(gene_ids_in_vocab >= 0)}/{len(gene_ids_in_vocab)} genes in vocabulary of size {len(vocab)}")

    # make sure all remaining genes are in vocabulary
    genes = adata.var["feature_name"].tolist()
    gene_ids = np.array(vocab(genes), dtype=int)
    assert np.all(gene_ids == np.array(adata.var["id_in_vocab"]))

    # get cell line and mean gene embeddings
    cell_embeddings, gene_embeddings = get_batch_embeddings(
        adata=adata,
        model=model.model,
        vocab=vocab,
        gene_ids=gene_ids,
        model_cfg=model_config,
        collator_cfg=collator_config,
        batch_size=16,
        max_length=8192,
        return_gene_embeddings=True,
    )

    # save cell line embeddings
    outpath = os.path.join(base_path, f"cell-embs/{model_name}.h5ad")
    cl_embs = ad.AnnData(X=cell_embeddings, obs=adata.obs, var=pd.DataFrame({"dim": [str(i) for i in range(cell_embeddings.shape[1])]}).set_index("dim"))
    cl_embs.write_h5ad(outpath)
    log.info(f"saved cell line embeddings to {outpath}")

    # handle genes with NaN embeddings
    nan_genes = np.where(np.any(np.isnan(gene_embeddings), axis=-1))[0]
    log.info(f"found {len(nan_genes)} genes with NaN embeddings, replacing with context-free embeddings")
    gene_embeddings[nan_genes] = gene_embeddings_ctx_free[nan_genes]

    # save NPZ of gene embeddings
    outpath = os.path.join(base_path, f"gene-embs/npz/{model_name}-gene-embs.npz")
    np.savez(
        outpath,
        gene_embeddings=gene_embeddings,
        genes_not_expressed=nan_genes,
        gene_embeddings_context_free=gene_embeddings_ctx_free,
        gene_names=list(gene2idx.keys()),
        gene_ids=list(gene2idx.values()),
    )
    log.info(f"saved gene embeddings to {outpath}")

    # process into AnnData for training
    process_mean_gene_embs(base_path, model_name)

    # reprocess AnnData of CCLE counts for contextual gene embeddings
    adata = sc.read_h5ad(os.path.join(base_path, "counts.h5ad"))
    gene_info_df = pd.read_csv(os.path.join(base_path, "raw/scgpt-genes.csv"))
    scgpt_gene_mapping = {key: value for (key, value) in zip(gene_info_df["feature_id"], gene_info_df["feature_name"])}
    scgpt_vocab_ids = set(gene_info_df["feature_id"])
    adata.var["in_scgpt_vocab"] = adata.var.index.map(lambda x: x in scgpt_vocab_ids)
    adata = adata[:, adata.var["in_scgpt_vocab"] == True]
    adata.var["gene_name_scgpt"] = adata.var.index.map(lambda gene_id: scgpt_gene_mapping[gene_id])
    adata.var = adata.var.drop(columns=["in_scgpt_vocab"])
    adata.layers["counts"] = adata.X.copy()
    log.info("processed CCLE AnnData for contextual gene embeddings")

    # subset to available genes
    adata.var["id_in_vocab"] = [vocab[gene] if gene in vocab else -1 for gene in adata.var["gene_name_scgpt"]]
    gene_ids_in_vocab = np.array(adata.var["id_in_vocab"])
    adata = adata[:, adata.var["id_in_vocab"] >= 0]
    log.info(f"matched {np.sum(gene_ids_in_vocab >= 0)}/{len(gene_ids_in_vocab)} genes in vocabulary of size {len(vocab)}")

    # get gene IDs
    genes = adata.var["gene_name_scgpt"].tolist()
    gene_ids = np.array(vocab(genes), dtype=int)

    # get count matrix
    count_matrix = adata.X
    count_matrix = (count_matrix if isinstance(count_matrix, np.ndarray) else count_matrix.A)

    # verify gene IDs
    if gene_ids is None:
        gene_ids = np.array(adata.var["id_in_vocab"])
        assert np.all(gene_ids >= 0)

    # set up dataset
    dataset = CountDataset(count_matrix, gene_ids, cls_token_id=vocab["<cls>"], pad_value=collator_config["pad_value"])

    # set up collator
    max_length = len(gene_ids)
    collator = DataCollator(
        do_padding=collator_config.get("do_padding", True),
        pad_token_id=collator_config.pad_token_id,
        pad_value=collator_config.pad_value,
        do_mlm=False,
        do_binning=collator_config.get("do_binning", True),
        mlm_probability=collator_config.mlm_probability,
        mask_value=collator_config.mask_value,
        max_length=max_length,
        sampling=False,
        data_style="pcpt",
        num_bins=collator_config.get("num_bins", 51),
        right_binning=collator_config.get("right_binning", False),
    )

    # set up data loader
    batch_size = 8
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=torch.utils.data.SequentialSampler(dataset),
        collate_fn=collator,
        drop_last=False,
        num_workers=min(len(os.sched_getaffinity(0)), batch_size),
        pin_memory=True,
    )

    # get lists for indexing
    cell_lines = adata.obs["ModelID"].tolist()
    genes = vocab.get_itos()

    # make empty objects to fill
    labels = []
    embeddings = []
    with torch.no_grad(), torch.cuda.amp.autocast(enabled=True, dtype=torch.bfloat16):

        # keep track of cell line
        count = 0
        pbar = tqdm(total=len(dataset))

        # iterate through data loader
        for data_dict in data_loader:

            # get batch embeddings
            input_gene_ids = data_dict["gene"].to(device)
            src_key_padding_mask = ~input_gene_ids.eq(collator_config["pad_token_id"])
            batch_embeddings = model.model._encode(
                src=input_gene_ids,
                values=data_dict["expr"].to(device),
                src_key_padding_mask=src_key_padding_mask,
            )

            # bring back to CPU
            input_gene_ids = input_gene_ids.cpu().numpy()
            batch_embeddings = batch_embeddings.to("cpu").to(torch.float32).numpy()

            # iterate through cell lines
            for i in range(batch_embeddings.shape[0]):

                # get cell line
                cell_line = cell_lines[count]
                cell_line_inputs = input_gene_ids[i]
                cell_line_embs = batch_embeddings[i]

                # iterate over genes
                for j in range(cell_line_embs.shape[0]):

                    # check if this is a real gene
                    input_id = cell_line_inputs[j]
                    if input_id in (60694, 60695, 60696):
                        continue

                    # fill embedding and label
                    labels.append(f"{cell_line} | {genes[input_id]}")
                    embeddings.append(cell_line_embs[j])

                # increment cell line
                count += 1
                pbar.update(1)

    # convert to arrays and normalize embeddings
    log.info("stacking and normalizing embeddings")
    labels = np.array(labels)
    embeddings = np.vstack(embeddings)
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

    # sort by label order
    log.info("sorting embeddings")
    sort_idx = np.argsort(labels)
    labels = labels[sort_idx]
    embeddings = embeddings[sort_idx]

    # process and save contextual gene embeddings
    utils.process_contextual_gene_embs(base_path, log, labels, embeddings, "_".join(model_name.split('-')))

if __name__ == "__main__":
    
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-path", type=str, required=True, help="Path to DepMap benchmark base directory.")
    parser.add_argument("--model-path", type=str, required=True, help="Path to scGPT model folder.")
    parser.add_argument("--model-name", type=str, required=True, help="Model name (filenames are based on this).")
    args = parser.parse_args()

    # run main function
    main(args.base_path, args.model_path, args.model_name)