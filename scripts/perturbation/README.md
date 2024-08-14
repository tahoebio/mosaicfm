### Steps to fine-tune

1. Prepare data first by running

```
python scripts/perturbation/prepare_data.py --data_path "/vevo/datasets/perturbation_datasets/" --dataset_name "norman" --vocab_path "/vevo/scgpt/checkpoints/release/scgpt-70m-1024-fix-norm-apr24-data/vocab.json"
```

2. Change the config and run finetune.py with composer

```
composer scripts/perturbation/finetune.py runai/config_perts/runai_finetune_70m_adamson.yaml 
```


