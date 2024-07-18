import argparse
import pandas as pd
import os


parser = argparse.ArgumentParser()
parser.add_argument("--log_path", required=True, help="path to the raw wandb log.")
parser.add_argument("--save_path", required=True, help="path to save the preprocessed log file.")

args = parser.parse_args()

df = pd.read_csv(args.log_path)

#Flops of each model for bath-size = 1
model_flops = {'scgpt-test-9m-full-data': 6.5E+10, 
            'scgpt-test-9m-0.1data': 6.5E+10,
            'scgpt-test-9m-0.01data': 6.5E+10,
            'scgpt-25m-1024-fix-norm-apr24-data': 1.0E+11, 
            'scgpt-70m-1024-fix-norm-apr24-data': 1.208E+12,
            'scgpt-70m-0.1xdata': 1.208E+12,
            'scgpt-70m-0.01xdata': 1.208E+12,
            'scgpt-70m-0.005xdata': 1.208E+12,
            'scgpt-1_3b-2048-prod': 1.897E+13 }

# batch-size of each model during training
model_batch_sizes = {'scgpt-test-9m-full-data': 16000, 
            'scgpt-test-9m-0.1data': 16000,
            'scgpt-test-9m-0.01data': 16000,
            'scgpt-25m-1024-fix-norm-apr24-data': 3200, 
            'scgpt-70m-1024-fix-norm-apr24-data': 4800,
            'scgpt-70m-0.1xdata': 4800,
            'scgpt-70m-0.01xdata': 4800,
            'scgpt-70m-0.005xdata': 4800,
            'scgpt-1_3b-2048-prod': 1792}

model_flops_per_batch = {}
for model in model_flops.keys():
    model_flops_per_batch[model] = model_flops[model] * model_batch_sizes[model]

# Adding columns for each model present in the dataframe
for column in df.columns:
    if '- _step' in column:
        model_name = column.replace(' - _step', '')
        if model_name in model_flops_per_batch:
            df[f'{model_name} - total-flops'] = df[column] * model_flops_per_batch[model_name]
    if 'MIN' in column or "MAX" in column:
        df = df.drop(column, axis=1)

df.to_csv(os.path.join(args.save_path, "processed_log.csv"), index=False)