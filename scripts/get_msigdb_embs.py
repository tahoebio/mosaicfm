import logging
import os
import argparse

import numpy as np
import torch
from omegaconf import OmegaConf as om
from scgpt.model import ComposerSCGPTModel
from scgpt.tokenizer import GeneVocab


def generate_embeddings(releases_path, save_path):
    log = logging.getLogger(__name__)
    logging.basicConfig(
        # Example of format string
        # 2022-06-29 11:22:26,152: [822018][MainThread]: INFO: Message here
        format=f"%(asctime)s: [%(process)d][%(threadName)s]: %(levelname)s: %(name)s: %(message)s"
    )
    logging.getLogger(__name__).setLevel("INFO")

    model_names = os.listdir(releases_path) 
    for model_name in model_names:
        model_dir = os.path.join(releases_path, model_name)

        model_config_path = os.path.join(model_dir, "model_config.yml")
        vocab_path = os.path.join(model_dir, "vocab.json")
        collator_config_path = os.path.join(model_dir, "collator_config.yml")
        model_file = os.path.join(model_dir, "best-model.pt")
        model_config = om.load(model_config_path)
        collator_config = om.load(collator_config_path)
        vocab = GeneVocab.from_file(vocab_path)

        model = ComposerSCGPTModel(
            model_config=model_config, collator_config=collator_config
        )
        model.load_state_dict(torch.load(model_file)["state"]["model"])
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.eval()
        log.info(f"Model loaded from {model_file}")

        with torch.no_grad(), torch.cuda.amp.autocast(enabled=True, dtype=torch.bfloat16):
            gene2idx = vocab.get_stoi()
            all_gene_ids = np.array([[id for id in gene2idx.values()]])
            chunk_size = 30000  # Size of each chunk, >30000 OOMs

            # Initialize an empty array to hold the final embeddings.
            # Assuming 'num_genes' is the total number of genes.
            # This should be equivalent to len(all_gene_ids.flatten()) in your case.
            num_genes = all_gene_ids.shape[1]
            gene_embeddings_ctx_free = (
                np.ones((num_genes, model_config["d_model"])) * np.nan
            )
            # Update output_size accordingly

            for i in range(0, num_genes, chunk_size):
                chunk_gene_ids = all_gene_ids[:, i : i + chunk_size]
                chunk_gene_ids_tensor = torch.tensor(chunk_gene_ids, dtype=torch.long).to(
                    device
                )

                token_embs = model.model.gene_encoder(chunk_gene_ids_tensor)
                flag_embs = model.model.flag_encoder(
                    torch.tensor(1, device=token_embs.device)
                ).expand(chunk_gene_ids_tensor.shape[0], chunk_gene_ids_tensor.shape[1], -1)

                total_embs = token_embs + flag_embs
                chunk_embeddings = model.model.transformer_encoder(total_embs)
                chunk_embeddings_cpu = chunk_embeddings.to("cpu").to(torch.float32).numpy()

                # Assigning the chunk embeddings to the correct place in the full array.
                gene_embeddings_ctx_free[i : i + chunk_size] = chunk_embeddings_cpu

            gene_embeddings_ctx_free_old = model.model.gene_encoder(
                torch.tensor(all_gene_ids, dtype=torch.long).to(device)
            )
            gene_embeddings_ctx_free_old = (
                gene_embeddings_ctx_free_old.to("cpu").to(torch.float32).numpy()
            )
            gene_embeddings_ctx_free_old = gene_embeddings_ctx_free_old[0, :, :]
        torch.cuda.empty_cache()
        log.info("Context free embeddings created.")
        gene_emb_save_path = os.path.join(save_path, f"gene_embeddings_{model_name}.npz") 
        np.savez(
            gene_emb_save_path,
            gene_embeddings_context_free=gene_embeddings_ctx_free,
            gene_embeddings_context_free_old=gene_embeddings_ctx_free_old,
            gene_names=list(gene2idx.keys()),
            gene_ids=list(gene2idx.values()),
        )
        log.info(f"Saved gene embeddings to {gene_emb_save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Embeddings for MsigDB")
    parser.add_argument("--releases_path", required=True, help="Path to where the checkpoints(model releases) are saved.") # e.g "/vevo/scgpt/checkpoints/release/"
    parser.add_argument("--save_path", required=True, help="Path to where to save the gene embeddings.") #e.g: /vevo/cellxgene/msigdb_gene_emb_subset/gene_embeddings_new/

    args = parser.parse_args()

    generate_embeddings(args.releases_path, args.save_path)