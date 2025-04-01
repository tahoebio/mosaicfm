# Protein Embedding Pipeline

A pipeline for generating protein embeddings using ESM-C models.

## Overview

This repo contains scripts for extracting protein sequences from UniProt and generating embeddings using EvolutionaryScale's ESM-C models. The purpose is to use these embeddings to initialize our gene representation model rather than using random initialization.

## Data Source

We're using UniProt data with the following specifications:
- Source: [UniProt KB (Reviewed + Ensembl)](https://www.uniprot.org/uniprotkb?query=%28database%3AEnsembl%29+AND+%28reviewed%3Atrue%29&facets=model_organism%3A9606)
- File: `uniprotkb_database_Ensembl_AND_reviewed_2025_03_22.tsv.gz`
- Content: Reviewed proteins with Ensembl gene mappings
- Format: Tab-separated values (TSV) with the following columns:
  - Entry (UniProt ID)
  - Gene Names (primary)
  - Entry Name
  - Protein names
  - Gene Names
  - Ensembl GeneID
  - Organism
  - Reviewed
  - Length
  - Sequence

## Repository Structure

```
protein-embedding-pipeline/
├── README.md
├── data/
│   └── uniprotkb_database_Ensembl_AND_reviewed_2025_03_22.tsv.gz
├── scripts/
│   ├── preprocess_uniprot.py     # Preprocess the UniProt data
│   └── generate_embeddings.py    # Generate protein embeddings
├── notebooks/
│   ├── explore_embeddings.ipynb  # Explore specific gene embeddings
│   └── embedding_clusters.ipynb  # UMAP and t-SNE visualizations
└── output/
    ├── human_proteins_processed.tsv
    └── protein_embeddings.h5
```

## Installation

Create a micromamba environment with required packages:

```bash
micromamba create -n esm-c python=3.10
micromamba activate esm-c

# Basic requirements
pip install torch pandas h5py tqdm matplotlib seaborn numpy huggingface_hub
# For speed improvements on GPUs
pip install flash-attn --no-build-isolation
# For visualizations in notebooks
pip install jupyter umap-learn scikit-learn

# Clone the ESM repo and install
git clone https://github.com/evolutionaryscale/esm.git
cd esm
pip install -e .
```

## Usage

### 1. Preprocess the UniProt data

```bash
python scripts/preprocess_uniprot.py
```

This converts the UniProt TSV to a processed TSV file, cleaning up column names and extracting relevant fields.

### 2. Generate embeddings

```bash
# Basic usage with the 300M model
python scripts/generate_embeddings.py

# With Flash Attention enabled
python scripts/generate_embeddings.py --use_flash_attn

# Using the larger 600M model
python scripts/generate_embeddings.py --model esmc_600m
```

This will:
- Process protein sequences from the preprocessed data
- Generate mean-pooled embeddings using the selected ESM-C model
- Save the embeddings to an HDF5 file with associated metadata

### 3. Explore the embeddings

Run the Jupyter notebooks in the `notebooks/` directory to:
- Explore specific genes of interest in `explore_embeddings.ipynb`
- Visualize embedding clusters using UMAP and t-SNE in `embedding_clusters.ipynb`

## Output Format

The output HDF5 file contains:
- A matrix of protein embeddings (num_proteins × embedding_dim)
- Metadata including UniProt IDs, gene names, Ensembl IDs, sequence lengths

### HDF5 File Structure

```
protein_embeddings.h5
├── embeddings               # Dataset [num_proteins × embedding_dim]
├── uniprot_ids              # Dataset [num_proteins]
├── gene_names               # Dataset [num_proteins]
├── ensembl_ids              # Dataset [num_proteins]
├── sequence_lengths         # Dataset [num_proteins]
├── sequences                # Dataset [num_proteins]
└── Attributes
    ├── num_proteins         # Total number of proteins
    ├── embedding_model      # Model used (esmc_300m or esmc_600m)
    ├── embedding_dimension  # Dimension of embeddings (1024 or 1280)
    ├── embedding_type       # "mean_pooled"
    └── flash_attention      # Whether Flash Attention was used
```

---

Created by Hamed Heydari @ Vevo  
April 01, 2024