# Copyright (C) Vevo Therapeutics 2025. All rights reserved.
import json
import logging
import os
import sys
from typing import Tuple

import numpy as np
import scanpy as sc
import torch
from omegaconf import DictConfig
from omegaconf import OmegaConf as om

from mosaicfm.model import ComposerSCGPTModel
from mosaicfm.tasks import get_batch_embeddings
from mosaicfm.tokenizer import GeneVocab

# Logging setup
logging.basicConfig(
    format="%(asctime)s: [%(process)d][%(threadName)s]: %(levelname)s: %(name)s: %(message)s",
    level=logging.INFO,
)
log = logging.getLogger(__name__)


def load_model(
    model_dir: str,
) -> Tuple[ComposerSCGPTModel, GeneVocab, DictConfig, DictConfig]:
    model_config_path = os.path.join(model_dir, "model_config.yml")
    vocab_path = os.path.join(model_dir, "vocab.json")
    collator_config_path = os.path.join(model_dir, "collator_config.yml")
    model_file = os.path.join(model_dir, "best-model.pt")

    model_config = om.load(model_config_path)
    collator_config = om.load(collator_config_path)
    vocab = GeneVocab.from_file(vocab_path)

    model = ComposerSCGPTModel(
        model_config=model_config,
        collator_config=collator_config,
    )
    model.load_state_dict(torch.load(model_file, weights_only=False)["state"]["model"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    log.info(f"Model loaded from {model_file}")
    return model, vocab, model_config, collator_config


def get_context_free_embeddings(
    model,
    vocab,
    d_model,
    device,
) -> Tuple[np.ndarray, np.ndarray]:
    gene2idx = vocab.get_stoi()
    all_gene_ids = np.array([list(gene2idx.values())])
    chunk_size = 30000
    num_genes = all_gene_ids.shape[1]
    gene_embeddings_te = np.ones((num_genes, d_model)) * np.nan

    with torch.no_grad(), torch.amp.autocast(
        "cuda",
        enabled=True,
        dtype=torch.bfloat16,
    ):
        for i in range(0, num_genes, chunk_size):
            chunk = all_gene_ids[:, i : i + chunk_size]
            chunk_tensor = torch.tensor(chunk, dtype=torch.long).to(device)

            token_embs = model.model.gene_encoder(chunk_tensor)
            flag_embs = model.model.flag_encoder(
                torch.tensor(1, device=device),
            ).expand_as(token_embs)
            total_embs = token_embs + flag_embs
            chunk_embs = model.model.transformer_encoder(total_embs)
            gene_embeddings_te[i : i + chunk_size] = chunk_embs.cpu().float().numpy()

        gene_embeddings_ge = model.model.gene_encoder(
            torch.tensor(all_gene_ids, dtype=torch.long).to(device),
        )
        gene_embeddings_ge = gene_embeddings_ge.to("cpu").to(torch.float32).numpy()
        gene_embeddings_ge = gene_embeddings_ge[0, :, :]

    return gene_embeddings_te, gene_embeddings_ge


def get_contextual_embeddings(
    cfg,
    model,
    vocab,
    model_config,
    collator_config,
    device,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, list]:
    adata = sc.read_h5ad(cfg.input_path)
    log.info(
        f"Loaded {adata.shape[0]} cells and {adata.shape[1]} genes from {cfg.input_path}",
    )

    if cfg.get("n_hvg"):
        sc.pp.highly_variable_genes(adata, n_top_genes=cfg.n_hvg, flavor="seurat_v3")
        adata = adata[:, adata.var["highly_variable"]]
        log.info(f"Performed HVG selection with n_top_genes = {cfg.n_hvg}")

    sc.pp.filter_cells(adata, min_genes=3)
    gene_col = cfg.get("gene_col", "feature_name")
    adata.var["id_in_vocab"] = [
        vocab[gene] if gene in vocab else -1 for gene in adata.var[gene_col]
    ]
    gene_ids_in_vocab = np.array(adata.var["id_in_vocab"])
    logging.info(
        f"match {np.sum(gene_ids_in_vocab >= 0)}/{len(gene_ids_in_vocab)} genes "
        f"in vocabulary of size {len(vocab)}.",
    )
    genes = adata.var[gene_col].tolist()
    gene_ids = np.array([vocab[gene] for gene in genes], dtype=int)

    cell_embs, gene_embs = get_batch_embeddings(
        adata=adata,
        model=model.model,
        vocab=vocab,
        gene_ids=gene_ids,
        model_cfg=model_config,
        collator_cfg=collator_config,
        batch_size=cfg.get("batch_size", 32),
        max_length=cfg.get("max_length", 8192),
        return_gene_embeddings=True,
    )
    return cell_embs, gene_embs, gene_ids, genes


def main(cfg: DictConfig):
    model, vocab, model_config, collator_config = load_model(cfg.model_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_name = cfg.get("model_name") or os.path.basename(
        os.path.normpath(cfg.model_dir),
    )

    gene_embeddings_te, gene_embeddings_ge = get_context_free_embeddings(
        model,
        vocab,
        model_config.d_model,
        device,
    )
    cell_embs, gene_embs_ea, gene_ids, genes_ensembl = get_contextual_embeddings(
        cfg,
        model,
        vocab,
        model_config,
        collator_config,
        device,
    )

    nan_genes = np.where(np.isnan(gene_embs_ea).any(axis=1))[0]
    gene_embs_ea[nan_genes] = gene_embeddings_te[nan_genes]

    with open(cfg.ensembl_to_gene_symbol) as f:
        ensembl_to_gene_symbol = json.load(f)
    gene_names_hgnc = [ensembl_to_gene_symbol.get(eid, eid) for eid in genes_ensembl]

    np.savez(
        os.path.join(cfg.output_path, f"gene_embeddings_{model_name}.npz"),
        gene_embeddings_EA=gene_embs_ea,
        gene_embeddings_TE=gene_embeddings_te,
        gene_embeddings_GE=gene_embeddings_ge,
        genes_not_expressed=nan_genes,
        gene_names=gene_names_hgnc,
        gene_ids=gene_ids,
        ensembl_ids=genes_ensembl,
    )
    log.info("Saved gene embeddings.")

    if cfg.get("save_cell_embeddings", True):
        adata = sc.read_h5ad(cfg.input_path)
        adata.obsm[f"X_{model_name}"] = cell_embs
        adata.write_h5ad(
            os.path.join(cfg.output_npz_path, f"adata_{model_name}.h5ad"),
            compression="gzip",
        )
        log.info("Saved adata with cell embeddings.")


if __name__ == "__main__":
    yaml_path = sys.argv[1]
    with open(yaml_path) as f:
        cfg = om.load(f)
    om.resolve(cfg)
    main(cfg)
