# Dataset Generation

## Random Sampling
To create a randomly downsampled version of dataset you need to have its original version in the MDS format. If you don't have it, download the original dataset either by scripts/download_cellxgene.py or from s3. You can then use ./download_cellxgene_random_sample.sh to select random chunks as the downsampled version. 
I used [version "2024-04-29" of cellxgene](s3://vevo-ml-datasets/vevo-scgpt/datasets/cellxgene_primary_2024-04-29_MDS/) and randomly downsampled 10, 1 and 0.5 percent of that.

1. download original cellxgene in your pvc as MDS format: s3://vevo-ml-datasets/vevo-scgpt/datasets/cellxgene_primary_2024-04-29_MDS/
2. provide source_dir, dest_dir and desired percentage to run download_cellxgene_random_sample.py script
```
bash scaling_exps/download_cellxgene_random_sample.py /vevo/datasets/cellxgene_primary_2024-04-29_MDS/train /vevo/datasets/one_percent_cellxgene_primary_2024-04-29_MDS/train 1
```
The script first downloads the specified percentage of chunks, then merges them all  and creates an index file. You can find the generated index.json in your specified dest_dir.

3. Upload the randomly downsampled dataset to s3 with ``` aws sync ```. 
You can find the paths to all existing subsampled datasets in the pdf document.

 
## Biased Sampling
I followed Geneformer's low-diversity dataset (Fig2.b) setup which uses [Madissoon study]{https://www.nature.com/articles/s41588-022-01243-4} to create the downsampled dataset. This study selects 3 tissues and around ~239K cells. 

-   Steps to recreate the dataset:
1. run scripts/scaling_exps/download_cellxgene_biased_ds.py to download the specified subset of cellxgene. You need to provide the version. If you want to use another subset of cellxgene edit the dataset-ids in the code.
The script download the chunks in h5ad format. 

``` Example:
python scripts/scaling_exps/download_cellxgene_biased_ds.py --version 2024-07-01 --old_vocab /vevo/datasets/cellxgene_primary_2024-04-29_MDS/cellxgene_primary_2024-04-29_vocab.json --dest_dir /vevo/datasets/biased_ds_cellxgene_primary_2024-07-01/train
```

2. Run scripts/make_dataset.py to convert each h5ad chunks to in format
``` Example:
python scripts/make_dataset.py --adata_dir /vevo/datasets/biased_ds_cellxgene --vocab_path /vevo/datasets/biased_ds_cellxgene/train/cellxgene_primary_2024-07-01_vocab.json --output_dir /vevo/datasets/biased_ds_cellxgene_hf

```

3. Run scripts/concatenate_datasets.py to merge all .hf files into one dataset and create the train/test splits. 
``` Example:
python scripts/concatenate_datasets.py --path /vevo/datasets/biased_ds_cellxgene_hf --dataset_name biased_ds_cellxgene_primary_2024-07-01
```

4. Run scripts/generate_mds.py
This script goes through the train.dataset which is a .hf folder and goes through all .arrow files (dataset chunks) and creates a MDS dataset.
``` Example:
python scripts/generate_mds.py --out_root /vevo/datasets/biased_ds_cellxgene_primary_2024-07-01_MDS --dataset_root_path /vevo/datasets/biased_ds_cellxgene_hf --train_suffix /biased_ds_cellxgene_primary_2024-07-01_train.dataset
``` 
- Beware that ``` train_suffix ``` should always start with / and there should be no / at the end of ```dataset_root_path```

5. Upload the biased downsampled dataset to s3 with ``` aws sync ```.



# Plotting
All the functions for plotting the scaling-law results are inside scripts/scaling_exps/create_plots.py.

1. To start with plotting first you need to download the results of metrics/eval/spearman for the models that you want as the csv format (make sure to choose time/batch as the x-axis ssince the preprocessing scripts only work with that.) 

2. Then you need to preprocess the log files to add #Flops per model for each step. Use scripts/scaling_exps/preprocess_logs.py to do so.
``` Example:
python scripts/scaling_exps/preprocess_logs.py --log_path /vevo/gits/old-vevo-scGPT/logs/v4/wandb_export_2024-07-17T11_08_33.315-04_00.csv  --save_path /vevo/gits/old-vevo-scGPT/logs/v4/
```

3. Then run create_plots.py for generating the plots. 

There are five different plots that I used:
- Parameter scale-down plot: parameter_scaling() function
- Dataset scale-down: dataset_scaling() function
- Scaling law plot: spearman_flops() function
- Max Spearman per models: spearman_flops_last_points() function
- Optimized model for a fixed FLOP: spearman_fix_cost() function

Note that some of these functions need to have some specific models in the log file to function properly. 

``` Example:
python scripts/scaling_exps/create_plots.py --csv_path /vevo/gits/old-vevo-scGPT/logs/v4/processed_log.csv  --save_path /vevo/gits/old-vevo-scGPT/logs/v4/
```