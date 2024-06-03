import random
from typing import Dict, Union
from anndata import AnnData
import boto3
import logging
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import torch


def set_seed(seed):
    """set random seed."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # if n_gpu > 0:
    #     torch.cuda.manual_seed_all(seed)


def add_file_handler(logger: logging.Logger, log_file_path: Path):
    """
    Add a file handler to the logger.
    """
    h = logging.FileHandler(log_file_path)

    # format showing time, name, function, and message
    formatter = logging.Formatter(
        "%(asctime)s-%(name)s-%(levelname)s-%(funcName)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    h.setFormatter(formatter)
    h.setLevel(logger.level)
    logger.addHandler(h)


def download_file_from_s3_url(s3_url, local_file_path):
    """
    Downloads a file from an S3 URL to the specified local path.

    :param s3_url: S3 URL of the file in the format "s3://bucket_name/path/to/file".
    :param local_file_path: Local path where the file will be saved.
    :return: The local path to the downloaded file.
    """
    # Validate the S3 URL format
    assert s3_url.startswith("s3://"), "URL must start with 's3://'"

    # Parse the S3 URL
    parsed_url = urlparse(s3_url)
    assert parsed_url.scheme == "s3", "URL scheme must be 's3'"

    bucket_name = parsed_url.netloc
    s3_file_key = parsed_url.path.lstrip("/")

    # Ensure bucket name and file key are not empty
    assert bucket_name, "Bucket name cannot be empty"
    assert s3_file_key, "S3 file key cannot be empty"

    # Create an S3 client
    s3 = boto3.client("s3")

    try:
        # Download the file
        s3.download_file(bucket_name, s3_file_key, local_file_path)
        print(f"File downloaded successfully to {local_file_path}")
        return local_file_path
    except Exception as e:
        print(f"Error downloading the file from {s3_url}: {e}")
        return None


def map_raw_id_to_vocab_id(
    raw_ids: Union[np.ndarray, torch.Tensor],
    gene_ids: np.ndarray,
) -> Union[np.ndarray, torch.Tensor]:
    """
    Map some raw ids which are indices of the raw gene names to the indices of the

    Args:
        raw_ids: the raw ids to map
        gene_ids: the gene ids to map to
    """
    if isinstance(raw_ids, torch.Tensor):
        device = raw_ids.device
        dtype = raw_ids.dtype
        return_pt = True
        raw_ids = raw_ids.cpu().numpy()
    elif isinstance(raw_ids, np.ndarray):
        return_pt = False
        dtype = raw_ids.dtype
    else:
        raise ValueError(f"raw_ids must be either torch.Tensor or np.ndarray.")

    if raw_ids.ndim != 1:
        raise ValueError(f"raw_ids must be 1d, got {raw_ids.ndim}d.")

    if gene_ids.ndim != 1:
        raise ValueError(f"gene_ids must be 1d, got {gene_ids.ndim}d.")

    mapped_ids: np.ndarray = gene_ids[raw_ids]
    assert mapped_ids.shape == raw_ids.shape
    if return_pt:
        return torch.from_numpy(mapped_ids).type(dtype).to(device)
    return mapped_ids.astype(dtype)


def compute_perturbation_metrics(
    results: Dict,
    ctrl_adata: AnnData,
    non_zero_genes: bool = False,
    return_raw: bool = False,
) -> Dict:
    """
    Given results from a model run and the ground truth, compute metrics

    Args:
        results (:obj:`Dict`): The results from a model run
        ctrl_adata (:obj:`AnnData`): The adata of the control condtion
        non_zero_genes (:obj:`bool`, optional): Whether to only consider non-zero
            genes in the ground truth when computing metrics
        return_raw (:obj:`bool`, optional): Whether to return the raw metrics or
            the mean of the metrics. Default is False.

    Returns:
        :obj:`Dict`: The metrics computed
    """
    from scipy.stats import pearsonr

    # metrics:
    #   Pearson correlation of expression on all genes, on DE genes,
    #   Pearson correlation of expression change on all genes, on DE genes,

    metrics_across_genes = {
        "pearson": [],
        "pearson_de": [],
        "pearson_delta": [],
        "pearson_de_delta": [],
    }

    metrics_across_conditions = {
        "pearson": [],
        "pearson_delta": [],
    }

    conditions = np.unique(results["pert_cat"])
    assert not "ctrl" in conditions, "ctrl should not be in test conditions"
    condition2idx = {c: np.where(results["pert_cat"] == c)[0] for c in conditions}

    mean_ctrl = np.array(ctrl_adata.X.mean(0)).flatten()  # (n_genes,)
    assert ctrl_adata.X.max() <= 1000, "gene expression should be log transformed"

    true_perturbed = results["truth"]  # (n_cells, n_genes)
    assert true_perturbed.max() <= 1000, "gene expression should be log transformed"
    true_mean_perturbed_by_condition = np.array(
        [true_perturbed[condition2idx[c]].mean(0) for c in conditions]
    )  # (n_conditions, n_genes)
    true_mean_delta_by_condition = true_mean_perturbed_by_condition - mean_ctrl
    zero_rows = np.where(np.all(true_mean_perturbed_by_condition == 0, axis=1))[
        0
    ].tolist()
    zero_cols = np.where(np.all(true_mean_perturbed_by_condition == 0, axis=0))[
        0
    ].tolist()

    pred_perturbed = results["pred"]  # (n_cells, n_genes)
    pred_mean_perturbed_by_condition = np.array(
        [pred_perturbed[condition2idx[c]].mean(0) for c in conditions]
    )  # (n_conditions, n_genes)
    pred_mean_delta_by_condition = pred_mean_perturbed_by_condition - mean_ctrl

    def corr_over_genes(x, y, conditions, res_list, skip_rows=[], non_zero_mask=None):
        """compute pearson correlation over genes for each condition"""
        for i, c in enumerate(conditions):
            if i in skip_rows:
                continue
            x_, y_ = x[i], y[i]
            if non_zero_mask is not None:
                x_ = x_[non_zero_mask[i]]
                y_ = y_[non_zero_mask[i]]
            res_list.append(pearsonr(x_, y_)[0])

    corr_over_genes(
        true_mean_perturbed_by_condition,
        pred_mean_perturbed_by_condition,
        conditions,
        metrics_across_genes["pearson"],
        zero_rows,
        non_zero_mask=true_mean_perturbed_by_condition != 0 if non_zero_genes else None,
    )
    corr_over_genes(
        true_mean_delta_by_condition,
        pred_mean_delta_by_condition,
        conditions,
        metrics_across_genes["pearson_delta"],
        zero_rows,
        non_zero_mask=true_mean_perturbed_by_condition != 0 if non_zero_genes else None,
    )

    def find_DE_genes(adata, condition, geneid2idx, non_zero_genes=False, top_n=20):
        """
        Find the DE genes for a condition
        """
        key_components = next(
            iter(adata.uns["rank_genes_groups_cov_all"].keys())
        ).split("_")
        assert len(key_components) == 3, "rank_genes_groups_cov_all key is not valid"

        condition_key = "_".join([key_components[0], condition, key_components[2]])

        de_genes = adata.uns["rank_genes_groups_cov_all"][condition_key]
        if non_zero_genes:
            de_genes = adata.uns["top_non_dropout_de_20"][condition_key]
            # de_genes = adata.uns["rank_genes_groups_cov_all"][condition_key]
            # de_genes = de_genes[adata.uns["non_zeros_gene_idx"][condition_key]]
            # assert len(de_genes) > top_n

        de_genes = de_genes[:top_n]

        de_idx = [geneid2idx[i] for i in de_genes]

        return de_idx, de_genes

    geneid2idx = dict(zip(ctrl_adata.var.index.values, range(len(ctrl_adata.var))))
    de_idx = {
        c: find_DE_genes(ctrl_adata, c, geneid2idx, non_zero_genes)[0]
        for c in conditions
    }
    mean_ctrl_de = np.array(
        [mean_ctrl[de_idx[c]] for c in conditions]
    )  # (n_conditions, n_diff_genes)

    true_mean_perturbed_by_condition_de = np.array(
        [
            true_mean_perturbed_by_condition[i, de_idx[c]]
            for i, c in enumerate(conditions)
        ]
    )  # (n_conditions, n_diff_genes)
    zero_rows_de = np.where(np.all(true_mean_perturbed_by_condition_de == 0, axis=1))[
        0
    ].tolist()
    true_mean_delta_by_condition_de = true_mean_perturbed_by_condition_de - mean_ctrl_de

    pred_mean_perturbed_by_condition_de = np.array(
        [
            pred_mean_perturbed_by_condition[i, de_idx[c]]
            for i, c in enumerate(conditions)
        ]
    )  # (n_conditions, n_diff_genes)
    pred_mean_delta_by_condition_de = pred_mean_perturbed_by_condition_de - mean_ctrl_de

    corr_over_genes(
        true_mean_perturbed_by_condition_de,
        pred_mean_perturbed_by_condition_de,
        conditions,
        metrics_across_genes["pearson_de"],
        zero_rows_de,
    )
    corr_over_genes(
        true_mean_delta_by_condition_de,
        pred_mean_delta_by_condition_de,
        conditions,
        metrics_across_genes["pearson_de_delta"],
        zero_rows_de,
    )

    if not return_raw:
        for k, v in metrics_across_genes.items():
            metrics_across_genes[k] = np.mean(v)
        for k, v in metrics_across_conditions.items():
            metrics_across_conditions[k] = np.mean(v)
    metrics = metrics_across_genes

    return metrics
