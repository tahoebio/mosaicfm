# Copyright (C) Vevo Therapeutics 2024. All rights reserved.
cd /src/vevo-scGPT
pip install -e .
cd scripts
composer train.py /src/vevo-scGPT/runai/scgpt-50m-train.yaml
```
