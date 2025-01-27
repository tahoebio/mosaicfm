# v2 Dataset Preparation Scripts

This folder contains scripts for the first major update to the pretraining dataset for MosaicFM.
This release includes data from CellXGene (~60M cells) as well as Vevo's Tahoe-100M dataset.


# Step 1: Update Vocab based on Tahoe data
```bash
python update_vocabulary.py cellxgene_2025_01_21.yaml
```
Note that for the new release, the vocabulary is keyed on ensembl ID instead of gene name.
We found that using the gene-names reported by cellxgene led to large mismatches when applied to other datasets, 
whereas the gene-IDs were more reliable.
For this release we use the Tahoe-100M dataset as the base and restrict cellxgene genes to the ones also included 
in Tahoe (which is almost all of them when keyed using gene-IDs).

# Step 2: Download and Prepare Datasets
## Step 2.1: Download CellXGene Data
```bash
python download_cellxgene.py cellxgene_2025_01_21.yaml
```

```bash
HF_HOME=<PATH ON PVC> python make_hf_dataset.py cellxgene_2025_01_21.yaml
```

Specifying the HF_HOME variable to be a path on PVC (such as "/vevo/cache") is necessary to ensure that the temporary 
cache doesn't blow up ephemeral storage when using a pod-based environment such as RunAI. 
Keep in mind that the memory usage of this script will keep growing up to 1TB and then stabilize around there.

## Step 2.2: Download and process scPerturb Data

[scPerturb](https://www.nature.com/articles/s41592-023-02144-y) is a collection of 44 single-cell perturbation datasets.
We first downloaded the collection of adata files from the scperturb Zenodo repository [here](https://zenodo.org/records/7041849) 
and then subsetted the data to only include genes that are in the vocabulary generated in Step 1. 
Since different datasets use different keys for indexing genes (e.g. gene names, ensembl IDs, etc.) 
or have data from non-human cells some of them are filtered out. 
We use a minimum filtering criteria that at least 60% of the genes in the dataset should be indexable to our vocabulary.
```bash
python process_scperturb.py scperturb.yaml
```

## Step 2.3: Download and process Vevo Data
For this release we used the portion of the Tahoe-100M dataset that passes "full" filters. 
For v1 of the dataset, we do not store any additional columns such as cell-line, plate or treatment information. 
These could be added in a future release if needed for model training. Furthermore, we do not aggregate the data based on 
any information about replication structure (eg: plate, batch ).

