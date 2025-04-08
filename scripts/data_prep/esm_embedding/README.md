# Protein Embedding Pipeline

A pipeline for generating and mixing protein embeddings using ESM-C model and integrating them with MosaicFM.

## Overview

This directory contains scripts for extracting protein sequences from UniProt, generating embeddings using EvolutionaryScale's ESM-C models, and mixing these embeddings with MosaicFM's gene representations. The purpose is to use these protein embeddings to initialize our gene representation model rather than using random initialization.

## Data Sources

- **UniProt:** [UniProt KB (Reviewed + Ensembl)](https://www.uniprot.org/uniprotkb?query=%28database%3AEnsembl%29+AND+%28reviewed%3Atrue%29&facets=model_organism%3A9606)
  - File: `uniprotkb_database_Ensembl_AND_reviewed_2025_03_22.tsv.gz`
  - Content: Reviewed proteins with Ensembl gene mappings
- **Ensembl Mapping:** `uniprot_to_ensembl.txt`
  - Maps UniProt IDs to Ensembl gene IDs

## Repository Structure

```
esm_embedding/
├── README.md
├── data/
│   ├── uniprotkb_database_Ensembl_AND_reviewed_2025_03_22.tsv.gz
│   └── uniprot_to_ensembl.txt
├── scripts/
│   ├── preprocess_uniprot.py     # Preprocess the UniProt data
│   ├── generate_embeddings.py    # Generate protein embeddings
│   └── mix_embedding.py          # Mix ESM-C embeddings with MosaicFM
├── notebooks/
│   ├── explore_embeddings.ipynb
│   ├── embedding_clusters.ipynb
│   └── gene_embeddings_mosaicfm-70m-tahoe.ipynb
├── output/                       # Files stored in S3://vevo-ml-datasets/esm-embeddings/
│   ├── human_proteins_processed.tsv
│   ├── protein_embeddings_esmc_600m.h5
│   ├── esmc_pretrained_data.pt
│   └── esmc_pretrained_gene_encoder.pt
└── log/
    └── mix_embeddings.out
```

## Installation

Create a micromamba environment with required packages:

```bash
micromamba create -n esm-c python=3.10
micromamba activate esm-c

# Basic requirements
pip install torch pandas h5py tqdm matplotlib seaborn numpy huggingface_hub boto
# For visualizations in notebooks
pip install jupyter umap-learn scikit-learn

# Clone the ESM repo and install
git clone https://github.com/evolutionaryscale/esm.git
cd esm
pip install -e .
```

## Pipeline Usage

### 1. Preprocess the UniProt data

```bash
python scripts/preprocess_uniprot.py
```

This converts the UniProt TSV to a processed TSV file, cleaning up column names and extracting relevant fields.

### 2. Generate ESM-C embeddings

```bash
# Using the 600M model (recommended)
python scripts/generate_embeddings.py --model esmc_600m
```

This generates mean-pooled embeddings from protein sequences and saves them to an HDF5 file.

### 3. Mix ESM-C embeddings with MosaicFM

```bash
python scripts/mix_embedding.py
```

This script:
- Loads ESM-C embeddings from the HDF5 file
- Loads the MosaicFM gene encoder
- Maps genes between UniProt and Ensembl IDs
- Replaces the MosaicFM gene embeddings with normalized ESM-C embeddings
- Saves the updated embeddings and metadata to PT files

## Output Files

### HDF5 File: `protein_embeddings_esmc_600m.h5`
ESM-C embeddings with metadata:
```
protein_embeddings_esmc_600m.h5
├── embeddings               # Dataset [19329 × 1152]
├── uniprot_ids              # Dataset [19329]
├── gene_names               # Dataset [19329]
├── ensembl_ids              # Dataset [19329]
├── sequence_lengths         # Dataset [19329]
├── sequences                # Dataset [19329]
└── Attributes
    ├── num_proteins         # 19329
    ├── embedding_model      # "esmc_600m"
    ├── embedding_dimension  # 1152
    ├── embedding_type       # "mean_pooled"
    └── flash_attention      # "enabled" or "disabled"
```

### PyTorch Files

#### `esmc_pretrained_data.pt`
Has the following structure:
```
esmc_pretrained_data.pt
├── embedding_matrix         # torch.Tensor [62720 × 1152]
├── gene_encoder_state_dict  # Dict containing model weights
├── vocab_size               # 62720
├── embedding_dim            # 1152
├── pad_token_id             # Padding token ID
├── use_norm                 # True
├── gene_names               # List of all tokens/gene names
├── replaced_indices         # List of vocab indices that were replaced
├── replaced_tokens          # List of tokens that were replaced
├── vocab_idx_to_esmc_idx    # Dict mapping vocab index to ESM-C index
├── total_matches            # 19183 (30.59% of vocab)
└── stats
    ├── orig_stats           # Original embedding statistics
    ├── esmc_stats           # ESM-C embedding statistics 
    └── new_stats            # New embedding statistics
```

#### `esmc_pretrained_gene_encoder.pt`
Just the gene encoder state dict for easy loading in MosaicFM models.

## Loading and Using PT Files

### Loading and exploring the complete data
```python
import torch
import boto3

# Load complete data
s3 = boto3.client('s3')
s3.download_file('vevo-ml-datasets', 'esm-embeddings/esmc_pretrained_data.pt', 'local_esmc_data.pt')
pretrained_data = torch.load('local_esmc_data.pt')

# Access embedding matrix
embedding_matrix = pretrained_data['embedding_matrix']
print(f"Embedding matrix shape: {embedding_matrix.shape}")  # [62720, 1152]

# Access metadata
vocab_size = pretrained_data['vocab_size']  # 62720
embedding_dim = pretrained_data['embedding_dim']  # 1152
pad_token_id = pretrained_data['pad_token_id']
use_norm = pretrained_data['use_norm']  # True

# Replacement statistics
replaced_indices = pretrained_data['replaced_indices']  # Indices of replaced embeddings
replaced_tokens = pretrained_data['replaced_tokens']  # Gene names that were replaced
total_matches = pretrained_data['total_matches']  # 19183
replacement_percentage = (total_matches / vocab_size) * 100  # 30.59%

# Access embedding statistics
orig_stats = pretrained_data['orig_stats']
new_stats = pretrained_data['new_stats']
print(f"Original norm: {orig_stats['norm']:.4f}, New norm: {new_stats['norm']:.4f}")

# Check if a specific gene was replaced
gene_of_interest = "ENSG00000197976"
gene_names = pretrained_data['gene_names']
if gene_of_interest in gene_names:
    idx = gene_names.index(gene_of_interest)
    is_replaced = idx in replaced_indices
    print(f"Gene {gene_of_interest} was {'replaced' if is_replaced else 'not replaced'}")
    
    if is_replaced:
        # Get the corresponding ESM-C index
        esmc_idx = pretrained_data['vocab_idx_to_esmc_idx'][idx]
        print(f"ESM-C index: {esmc_idx}")
```

### Loading just the gene encoder
```python
import torch
import boto3
import mosaicfm as mfm

# Create a GeneEncoder instance with the correct dimensions
encoder = mfm.model.blocks.GeneEncoder(
    vocab_size=62720,
    embedding_dim=1152,  # For ESM-C 600M
    padding_idx=0,
    use_norm=True,
)

# Load the pretrained weights
s3 = boto3.client('s3')
s3.download_file('vevo-ml-datasets', 'esm-embeddings/esmc_pretrained_gene_encoder.pt', 'local_gene_encoder.pt')
encoder.load_state_dict(torch.load('local_gene_encoder.pt'))

# Use in a model
# model.gene_encoder = encoder
```

## S3 Storage

The embeddings and processed files are stored in:
```
s3://vevo-ml-datasets/esm-embeddings/
```

---

Created by Hamed Heydari @ Vevo  
Updated: April 2025