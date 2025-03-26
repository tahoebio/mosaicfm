# imports
import sys
import glob
import argparse
import torch
import wandb
import numpy as np
import pandas as pd
import scanpy as sc
import torch.nn as nn
import torch.optim as optim
import lightning as L
import torchmetrics.functional as tmf
from torch.utils.data import DataLoader, Dataset, random_split
from sklearn.model_selection import train_test_split
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from tqdm import tqdm

# dataset for sensitivity prediction
class PRISMSensitivityDataset(Dataset):

    def __init__(self, dataframe, model_key):

        # save provided DataFrame
        self.df = dataframe

        # load embeddings and make ModelID --> row dictionary
        self.adata = sc.read_h5ad("/vevo/umair/data/sens-pred/embs/ccle.h5ad")
        self.cl_to_row = {cl: i for i, cl in enumerate(self.adata.obs["ModelID"].tolist())}

        # select emnbeddings
        self.model_key = model_key
        self.emb_array = self.adata.obsm[self.model_key]
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):

        # retrieve row
        row = self.df.iloc[idx]
        cell_line = row["cell_line"]

        # retrieve and assemble inputs
        cl_emb = torch.tensor(self.emb_array[self.cl_to_row[cell_line]], dtype=torch.float32)
        morgan_fp = torch.tensor(row["morgan_fp"], dtype=torch.float32)
        dosage = torch.tensor([row["dosage"]], dtype=torch.float32)
        features = torch.cat([cl_emb, morgan_fp, dosage])

        # retrieve target and return along with row index
        target = torch.tensor([row["growth_rate"]], dtype=torch.float32)
        return features, target

# define MLP model
class MLP(L.LightningModule):

    def __init__(self, input_dim, lr=5e-4):

        # required
        super().__init__()
        self.save_hyperparameters()

        # save learning rate
        self.lr = lr

        # use MSE loss
        self.loss_fn = nn.MSELoss()
        
        # define model architecture
        self.model = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.3),
            nn.Linear(128, 1)
        )
    
    def forward(self, x):
        return self.model(x).squeeze()
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        y, y_pred = y.squeeze(), y_pred.squeeze()
        loss = self.loss_fn(y_pred, y)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        y, y_pred = y.squeeze(), y_pred.squeeze()
        loss = self.loss_fn(y_pred, y)
        r2 = tmf.r2_score(y_pred, y)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_r2", r2, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss
    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

# define linear probe model
class LinearProbe(L.LightningModule):

    def __init__(self, input_dim, lr=5e-4):
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr
        self.loss_fn = nn.MSELoss()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        return self.linear(x).squeeze()

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        y, y_pred = y.squeeze(), y_pred.squeeze()
        loss = self.loss_fn(y_pred, y)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        y, y_pred = y.squeeze(), y_pred.squeeze()
        loss = self.loss_fn(y_pred, y)
        r2 = tmf.r2_score(y_pred, y)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_r2", r2, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

# split dataset by cell line
def split_by_cell_line(dataframe, val_ratio=0.2):
    unique_cell_lines = dataframe["cell_line"].unique()
    train_cell_lines, val_cell_lines = train_test_split(unique_cell_lines, test_size=val_ratio, random_state=42)
    train_df = dataframe[dataframe["cell_line"].isin(train_cell_lines)]
    val_df = dataframe[dataframe["cell_line"].isin(val_cell_lines)]
    return train_df, val_df

# split dataset by drug
def split_by_drug(dataframe, val_ratio=0.2):
    unique_drugs = dataframe["broad_id"].unique()
    train_drugs, val_drugs = train_test_split(unique_drugs, test_size=val_ratio, random_state=42)
    train_df = dataframe[dataframe["broad_id"].isin(train_drugs)]
    val_df = dataframe[dataframe["broad_id"].isin(val_drugs)]
    return train_df, val_df 

# split dataset by cell line and drug
def split_by_cl_and_drug(dataframe, val_ratio=0.2):
    unique_cell_lines = dataframe["cell_line"].unique()
    unique_drugs = dataframe["broad_id"].unique()
    train_cell_lines, val_cell_lines = train_test_split(unique_cell_lines, test_size=val_ratio, random_state=42)
    train_drugs, val_drugs = train_test_split(unique_drugs, test_size=val_ratio, random_state=42)
    train_df = dataframe[(dataframe["cell_line"].isin(train_cell_lines)) & (dataframe["broad_id"].isin(train_drugs))]
    val_df = dataframe[(dataframe["cell_line"].isin(val_cell_lines)) | (dataframe["broad_id"].isin(val_drugs))]
    return train_df, val_df

# dictionary of data split functions
SPLIT_FUNCS = {
    "cl": split_by_cell_line,
    "drug": split_by_drug,
    "cl+drug": split_by_cl_and_drug
}

# split data based on user specification
def prepare_data(split_str, model_name, val_split):

    if split_str == "random":

        # determine W&B name
        wandb_name = model_name

        # set up dataset
        df = pd.read_pickle("/vevo/umair/data/sens-pred/mlp-data/dataset.pkl")
        dataset = PRISMSensitivityDataset(df, model_name)
        sample_features, _ = dataset[0]
        input_dim = sample_features.shape[0]

        # split dataset
        total_size = len(dataset)
        val_size = int(total_size * val_split)
        train_size = total_size - val_size
        rng = torch.Generator().manual_seed(42)
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=rng)

    elif split_str in ("cl", "drug", "cl+drug"):

        # determine W&B name
        wandb_name = f"{model_name}_split_{split_str}"

        # set up datasets
        df = pd.read_pickle("/vevo/umair/data/sens-pred/mlp-data/dataset.pkl")
        train_df, val_df = SPLIT_FUNCS[split_str](df)
        train_dataset = PRISMSensitivityDataset(train_df, model_name)
        val_dataset = PRISMSensitivityDataset(val_df, model_name)
        sample_features, _ = train_dataset[0]
        input_dim = sample_features.shape[0]

    else:
        sys.exit("Unrecognized data split.")

    return wandb_name, train_dataset, val_dataset, input_dim

# get path to model checkpoint given W&B name
def find_checkpoint(wandb_name, model_dir="/vevo/umair/data/sens-pred/mlp-models"):
    matches = glob.glob(f"{model_dir}/{wandb_name}_epoch*.ckpt")
    if len(matches) != 1:
        sys.exit(f"Expected one matching checkpoint for '{wandb_name}', found {len(matches)}.")
    return matches[0]

# training script for given model
def train_model(
    model_name,
    split_str,
    architecture,
    batch_size=2048,
    epochs=20,
    val_split=0.2,
    save_dir="/vevo/umair/data/sens-pred/mlp-models"
):

    # header
    print(f"\n========== training for {model_name} | {split_str} split | {architecture}  ==========\n")

    # prepare data and make loaders
    wandb_name, train_dataset, val_dataset, input_dim = prepare_data(split_str, model_name, val_split)
    train_loader = DataLoader(train_dataset, num_workers=8, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, num_workers=8, batch_size=batch_size, shuffle=False)

    # initialize model
    if architecture == "mlp":
        model = MLP(input_dim)
    elif architecture == "probe":
        model = LinearProbe(input_dim)
        wandb_name = f"probe_{wandb_name}"
    else:
        sys.exit("Unrecognized architecture.")

    # initialize logging
    wandb.init(project="mlp-prism-sens", name=wandb_name, reinit=True)
    wandb_logger = WandbLogger(project="mlp-prism-sens", name=wandb_name)

    # set up checkpoints
    checkpoint_callback = ModelCheckpoint(
        dirpath=save_dir,
        filename=f"{wandb_name}_epoch-{{epoch:02d}}_val-loss-{{val_loss:.4f}}",
        save_top_k=1,
        monitor="val_loss",
        mode="min",
        auto_insert_metric_name=False
    )

    # set up trainer
    trainer = L.Trainer(
        max_epochs=epochs,
        logger=wandb_logger,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        callbacks=[checkpoint_callback]
    )

    # train model
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

# save predictions on validation set for given model# training script for given model
def save_model_predictions(
    model_name,
    split_str,
    architecture,
    batch_size=2048,
    val_split=0.2,
    save_dir="/vevo/umair/data/sens-pred/mlp-preds"
):

    # header
    print(f"\n========== saving predictions for {model_name} | {split_str} split | {architecture} ==========\n")

    # prepare data and make loaders
    wandb_name, _, val_dataset, _ = prepare_data(split_str, model_name, val_split)
    val_loader = DataLoader(val_dataset, num_workers=8, batch_size=batch_size, shuffle=False)

    # load model
    if architecture == "mlp":
        ckpt_path = find_checkpoint(wandb_name)
        model = MLP.load_from_checkpoint(ckpt_path)
    elif architecture == "probe":
        wandb_name = f"probe_{wandb_name}"
        ckpt_path = find_checkpoint(wandb_name)
        model = LinearProbe.load_from_checkpoint(ckpt_path)
    else:
        sys.exit("Unrecognized architecture.")

    # set up model
    model.eval()
    model = model.to("cuda")

    # iterate over validation set
    preds = []
    with torch.no_grad():
        for batch in tqdm(val_loader):
            x, _ = batch
            x = x.to(model.device)
            y_pred = model(x)
            preds.append(y_pred.cpu().numpy())

    # flatten and make dataframe to save
    preds = np.concatenate(preds)
    results = val_dataset.df[["broad_id", "dosage", "cell_line", "growth_rate"]].copy()
    results["predicted"] = preds

    # save results
    save_path = f"{save_dir}/{wandb_name}.csv"
    results.to_csv(save_path, index=False)
    print(f"Saved predictions to {save_path}.")

if __name__ == "__main__":

    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--split", type=str, required=True)
    parser.add_argument("--architecture", type=str, required=True)
    parser.add_argument("--task", type=str, required=True)
    args = parser.parse_args()

    # run appropriate function
    if args.task == "train":
        train_model(args.model, args.split, args.architecture)
    elif args.task == "predict":
        save_model_predictions(args.model, args.split, args.architecture)
    else:
        sys.exit("Unrecognized task.")