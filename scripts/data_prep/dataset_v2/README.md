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

# Step 2: Download and Prepare CellXGene data
```bash
python download_cellxgene.py cellxgene_2025_01_21.yaml
```

```bash
HF_HOME=<PATH ON PVC> python make_hf_dataset.py cellxgene_2025_01_21.yaml
```

Specifying the HF_HOME variable to be a path on PVC (such as "/vevo/cache") is necessary to ensure that the temporary 
cache doesn't blow up ephemeral storage when using a pod-based environment such as RunAI. 
Keep in mind that the memory usage of this script will keep growing up to 1TB and then stabilize around there.