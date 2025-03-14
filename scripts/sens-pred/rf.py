import os
import sys
import argparse
import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import roc_auc_score, average_precision_score
from tqdm import tqdm

# parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("--base-path", type=str, required=True)
parser.add_argument("--dataset", type=str, required=True)
parser.add_argument("--emb", type=str, required=True)
parser.add_argument("--problem", type=str, required=True)
parser.add_argument("--add-label", type=str, default="")
parser.add_argument("--n-jobs", type=int, default=16)
parser.add_argument("--n-permutations", type=int, default=1)
args = parser.parse_args()

# print header
print(f"\n===== dataset: {args.dataset} | emb: {args.emb} | problem: {args.problem} =====\n")

# load data
if args.dataset == "mosaic":
    sens = pd.read_csv(os.path.join(args.base_path, "sens-rif/growth-rates.csv"), index_col=0)
    embs = sc.read_h5ad(os.path.join(args.base_path, "embs/rif-dmso-t0-mean.h5ad"))
    obs_col = "cell-line"
elif args.dataset == "prism":
    sens = pd.read_csv(os.path.join(args.base_path, "sens-prism/logfold-changes.csv"))
    embs = sc.read_h5ad(os.path.join(args.base_path, "embs/ccle.h5ad"))
    obs_col = "ModelID"
elif args.dataset == "prism-sec":
    sens = pd.read_csv(os.path.join(args.base_path, "sens-prism-sec/logfold-changes.csv"))
    embs = sc.read_h5ad(os.path.join(args.base_path, "embs/ccle.h5ad"))
    obs_col = "ModelID"
else:
    sys.exit("unrecognized dataset (must be mosaic, prism, or prism-sec)")

# extract cell lines and conditions
cell_lines = sens["cell_line"].unique().tolist()
conditions = [c for c in sens["condition"].unique() if "DMSO" not in c]

# intialize results
if args.problem == "classification":
    results = {
        "condition": [],
        "embedding": [],
        "auroc": [],
        "auprc": [],
        "permuted": [],
        "permuted-iter": []
    }
else:
    results = {
        "condition": [],
        "embedding": [],
        "r2": [],
        "permuted": [],
        "permuted-iter": []
    }

# one model per condition
for condition in tqdm(conditions):

    # subset to valid labels
    sens_subset = sens[sens["condition"] == condition][["cell_line", "growth_rate"]]
    sens_subset = sens_subset[~sens_subset["growth_rate"].isna()]

    # restrict to available embeddings 
    adata = embs[embs.obs[obs_col].isin(sens_subset["cell_line"])]
    labels = adata.obs.merge(sens_subset, left_on=obs_col, right_on="cell_line")["growth_rate"].to_numpy()

    # skip conditions with no available embeddings
    if adata.shape[0] == 0:
        continue

    # discretize if needed
    if args.problem == "classification":
        labels = (labels < 0)

    # skip conditions where all growth rates are above or below zero
    if args.problem == "classification" and (np.all(labels) or np.all(~labels)):
        continue

    # train models
    if args.problem == "classification":
        rf = RandomForestClassifier(oob_score=True, n_jobs=args.n_jobs)
        rfs_permuted = [RandomForestClassifier(oob_score=True, n_jobs=args.n_jobs) for _ in range(args.n_permutations)]
    else:
        rf = RandomForestRegressor(oob_score=True, n_jobs=args.n_jobs)
        rfs_permuted = [RandomForestRegressor(oob_score=True, n_jobs=args.n_jobs) for _ in range(args.n_permutations)]
    rf.fit(adata.obsm[args.emb], labels)
    for i in range(args.n_permutations):
        rfs_permuted[i].fit(adata.obsm[args.emb], np.random.permutation(labels))

    # save results
    results["condition"].append(condition)
    results["embedding"].append(args.emb)
    results["permuted"].append("false")
    results["permuted-iter"].append(0)
    if args.problem == "classification":
        results["auroc"].append(roc_auc_score(labels, rf.oob_decision_function_[:, 1]))
        results["auprc"].append(average_precision_score(labels, rf.oob_decision_function_[:, 1]))
    else:
        results["r2"].append(rf.oob_score_)

    # save permuted results
    for i in range(args.n_permutations):
        results["condition"].append(condition)
        results["embedding"].append(args.emb)
        results["permuted"].append("true")
        results["permuted-iter"].append(i + 1)
        if args.problem == "classification":
            results["auroc"].append(roc_auc_score(labels, rfs_permuted[i].oob_decision_function_[:, 1]))
            results["auprc"].append(average_precision_score(labels, rfs_permuted[i].oob_decision_function_[:, 1]))
        else:
            results["r2"].append(rfs_permuted[i].oob_score_)

# convert to DataFrame and save
results = pd.DataFrame.from_dict(results)
outpath = os.path.join(args.base_path, f"results/{args.dataset}_{args.problem}_{args.emb}{args.add_label}.csv")
results.to_csv(outpath, index=False)
print(f"saved results to {outpath}")