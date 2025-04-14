# Copyright (C) Vevo Therapeutics 2025. All rights reserved.
import os

import h5py
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

import mosaicfm as mfm

model_paths = {
    "mosaicfm-70m-tahoe": "/vevo/model/release/mosaicfm-70m-tahoe/",
}

EMBEDDING_DIM = 1152  # ESM-C 600M
USE_NORM = True

# Select model
model_name = "mosaicfm-70m-tahoe"
vocab_path = os.path.join(model_paths[model_name], "vocab.json")

############################################################################
# Define Vocab
############################################################################
vocab = mfm.tokenizer.GeneVocab.from_file(vocab_path)

# Create DataFrame for vocabulary
vocab_df = pd.DataFrame(
    {
        "token": list(vocab.index_to_token.values()),
        "vocab_idx": list(range(len(vocab))),
    },
)
print(f"Vocabulary size: {len(vocab_df)}")

vocab_size = len(vocab)
pad_token_id = vocab["<pad>"]

############################################################################
# Define GeneEncoder
############################################################################
encoder = mfm.model.blocks.GeneEncoder(
    vocab_size,
    EMBEDDING_DIM,
    padding_idx=pad_token_id,
    use_norm=USE_NORM,
)

############################################################################
# Extract embedding
############################################################################
original_embedding = encoder.embedding.weight.clone()
print(f"Original embedding shape: {original_embedding.shape}")

############################################################################
# Load ESM-C embeddings
############################################################################
embedding_file = "../output/protein_embeddings_esmc_600m.h5"
print(f"Loading ESM-C embeddings from {embedding_file}")

with h5py.File(embedding_file, "r") as f:
    esmc_embeddings = f["embeddings"][:]
    uniprot_ids = [id.decode("utf-8") for id in f["uniprot_ids"][:]]
    gene_names_esmc = [
        name.decode("utf-8") if name else "" for name in f["gene_names"][:]
    ]

    print(
        f"Loaded {len(esmc_embeddings)} ESM-C embeddings with dimension {esmc_embeddings.shape[1]}",
    )

# Create DataFrame for ESM-C embeddings
esmc_df = pd.DataFrame(
    {
        "uniprot_id": uniprot_ids,
        "gene_name": gene_names_esmc,
        "esmc_idx": range(len(uniprot_ids)),  # Index into the embeddings array
    },
)

############################################################################
# Load and merge protein data
############################################################################
prot = pd.read_csv("../output/human_proteins_processed.tsv", sep="\t")
ensembl = pd.read_csv("../data/uniprot_to_ensembl.txt", sep="\t", header=None)
ensembl.columns = ["prot", "ensembl_id"]

# Remove duplicates from ensembl data
ensembl_dedup = ensembl.drop_duplicates(subset=["prot"])


# Create short ensembl IDs by removing version numbers
def remove_version(ensembl_id):
    if pd.isna(ensembl_id):
        return None
    return str(ensembl_id).split(".")[0]


# Apply the function to ensembl data
ensembl_dedup["ensembl_id_short"] = ensembl_dedup["ensembl_id"].apply(remove_version)

# Merge protein and ensembl data
merged_df = pd.merge(
    prot,
    ensembl_dedup,
    left_on=prot.columns[0],  # Assuming first column is protein ID
    right_on="prot",
    how="inner",
)

# Merge with ESM-C embedding data
merged_with_esmc = pd.merge(
    merged_df,
    esmc_df,
    left_on="prot",
    right_on="uniprot_id",
    how="inner",
)

print(f"Merged DataFrame with ESM-C: {len(merged_with_esmc)} rows")

############################################################################
# Map vocabulary to ESM-C embeddings
############################################################################
# Try to match by Ensembl ID first
all_mappings = pd.merge(
    vocab_df,
    merged_with_esmc[
        ["ensembl_id_short", "esmc_idx", "gene_names_primary"]
    ].drop_duplicates(),
    left_on="token",
    right_on="ensembl_id_short",
    how="inner",
)
print(f"Matched by Ensembl ID: {len(all_mappings)} tokens")

# Sort by vocabulary index for consistency
all_mappings = all_mappings.sort_values("vocab_idx").reset_index(drop=True)

############################################################################
# Replace embeddings
############################################################################
# Clone the original embedding
new_embedding = original_embedding.clone()

# Get statistics of original embedding
orig_mean = original_embedding.mean(dim=0).detach()
orig_std = original_embedding.std(dim=0).detach()
orig_norm = torch.norm(original_embedding, dim=1).mean().detach()
print(
    f"Original embedding stats - Mean: {orig_mean.mean().item():.4f}, Std: {orig_std.mean().item():.4f}, Norm: {orig_norm.item():.4f}",
)

# Get statistics of ESM-C embeddings to use for replacement
esmc_idxs_to_use = all_mappings["esmc_idx"].values
esmc_subset = esmc_embeddings[esmc_idxs_to_use]
esmc_mean = np.mean(esmc_subset, axis=0)
esmc_std = np.std(esmc_subset, axis=0)
esmc_norm = np.mean(np.linalg.norm(esmc_subset, axis=1))
print(
    f"ESM-C embedding stats - Mean: {np.mean(esmc_mean):.4f}, Std: {np.mean(esmc_std):.4f}, Norm: {esmc_norm:.4f}",
)

# Replace embeddings with normalization
for _, row in tqdm(all_mappings.iterrows(), desc="Replacing embeddings"):
    vocab_idx = row["vocab_idx"]
    esmc_idx = row["esmc_idx"]
    esmc_emb = esmc_embeddings[esmc_idx]

    # Normalize embedding to match original distribution
    normalized_emb = (esmc_emb - esmc_mean) / (
        esmc_std + 1e-8
    )  # Avoid division by zero
    normalized_emb = (
        normalized_emb * orig_std.detach().cpu().numpy()
        + orig_mean.detach().cpu().numpy()
    )

    # Replace embedding
    new_embedding[vocab_idx] = torch.tensor(
        normalized_emb,
        dtype=original_embedding.dtype,
    )

# Get statistics of new embedding
replaced_indices = all_mappings["vocab_idx"].values
new_mean = new_embedding[replaced_indices].mean(dim=0).detach()
new_std = new_embedding[replaced_indices].std(dim=0).detach()
new_norm = torch.norm(new_embedding[replaced_indices], dim=1).mean().detach()
print(
    f"New embedding stats - Mean: {new_mean.mean().item():.4f}, Std: {new_std.mean().item():.4f}, Norm: {new_norm.item():.4f}",
)

############################################################################
# Test embedding
############################################################################
# Create new encoder with replaced embeddings
new_encoder = mfm.model.blocks.GeneEncoder(
    vocab_size,
    EMBEDDING_DIM,
    padding_idx=pad_token_id,
    use_norm=USE_NORM,
)
new_encoder.embedding.weight.data = new_embedding

# Create lists for replaced indices and tokens
replaced_indices = all_mappings["vocab_idx"].tolist()
replaced_tokens = all_mappings["token"].tolist()

# Create mapping dictionary
vocab_idx_to_esmc_idx = all_mappings.set_index("vocab_idx")["esmc_idx"].to_dict()

# Save everything in a single file
pretrained_data = {
    # Embeddings and model
    "embedding_matrix": new_embedding,
    "gene_encoder_state_dict": new_encoder.state_dict(),
    # Vocabulary information
    "vocab_size": vocab_size,
    "embedding_dim": EMBEDDING_DIM,
    "pad_token_id": pad_token_id,
    "use_norm": USE_NORM,
    "gene_names": vocab_df["token"].tolist(),
    # Replacement info
    "replaced_indices": replaced_indices,  # Vocab indices that were replaced
    "replaced_tokens": replaced_tokens,  # Corresponding tokens
    "vocab_idx_to_esmc_idx": vocab_idx_to_esmc_idx,  # Map from vocab index to ESM-C index
    # Mapping data
    "total_matches": len(all_mappings),
    # Original and new embedding stats
    "orig_stats": {
        "mean": orig_mean.mean().item(),
        "std": orig_std.mean().item(),
        "norm": orig_norm.item(),
    },
    "esmc_stats": {
        "mean": float(np.mean(esmc_mean)),
        "std": float(np.mean(esmc_std)),
        "norm": float(esmc_norm),
    },
    "new_stats": {
        "mean": new_mean.mean().item(),
        "std": new_std.mean().item(),
        "norm": new_norm.item(),
    },
}

# Save the combined data
torch.save(pretrained_data, "../output/esmc_pretrained_data.pt")

# Also save just the gene encoder for easy loading
torch.save(new_encoder.state_dict(), "../output/esmc_pretrained_gene_encoder.pt")

print("Saved all data to esmc_pretrained_data.pt")
print(
    f"Successfully replaced {len(all_mappings)} out of {len(vocab_df)} embeddings ({len(all_mappings)/len(vocab_df)*100:.2f}%)",
)

# Verification code for a sample token
print("\nVerification for a sample token:")
sample_token = "ENSG00000197976"
sample_row = vocab_df[vocab_df["token"] == sample_token]

if not sample_row.empty:
    vocab_idx = sample_row.iloc[0]["vocab_idx"]
    print(f"Token '{sample_token}' is at index {vocab_idx} in vocabulary")

    # Check if this token was mapped
    mapping_row = all_mappings[all_mappings["vocab_idx"] == vocab_idx]
    if not mapping_row.empty:
        esmc_idx = mapping_row.iloc[0]["esmc_idx"]
        print(f"This was replaced with ESM-C embedding at index {esmc_idx}")

        if (
            "ensembl_id_short" in mapping_row.columns
            and mapping_row.iloc[0]["ensembl_id_short"] == sample_token
        ):
            print("  Matched as Ensembl ID")
        elif (
            "gene_names_primary" in mapping_row.columns
            and mapping_row.iloc[0]["gene_names_primary"] == sample_token
        ):
            print("  Matched as gene name")

        # Find the corresponding protein
        protein_row = merged_with_esmc[merged_with_esmc["esmc_idx"] == esmc_idx].iloc[0]
        print(f"  Associated protein: {protein_row['prot']}")
        print(f"  Associated gene: {protein_row['gene_names_primary']}")
    else:
        print("This token was not replaced with an ESM-C embedding")
else:
    print(f"Token '{sample_token}' not found in vocabulary")
