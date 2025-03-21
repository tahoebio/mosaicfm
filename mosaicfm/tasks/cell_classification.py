# Copyright (C) Vevo Therapeutics 2025. All rights reserved.
import json

import numpy as np
import scanpy as sc
from composer import State
from composer.core.callback import Callback
from composer.loggers import Logger
from sklearn.linear_model import LogisticRegression

from mosaicfm.model import ComposerSCGPTModel
from mosaicfm.tasks import get_batch_embeddings
from mosaicfm.tokenizer import GeneVocab


# Custom Callback to run the cell classification after training
class CellClassification(Callback):
    def __init__(
        self,
        ensemble_to_gene_path: str,
        model: ComposerSCGPTModel,
        vocab: GeneVocab,
        model_config,
        collator_config,
    ):

        super().__init__()

        # load gene_to_id mapping
        with open(ensemble_to_gene_path) as f:
            id_to_gene = json.load(f)
        self.gene_to_id = dict(zip(id_to_gene.values(), id_to_gene.keys()))

        model.eval()
        self.model = model
        self.vocab = vocab
        model_config["precision"] = "amp_bf16"
        self.model_config = model_config

        self.collator_config = collator_config
        self.dataset_registry = {
            "zheng": {
                "train": "/vevo/gits/mosaicfm/debug_notebooks/data/zheng/zheng_train.h5ad",
                "test": "/vevo/gits/mosaicfm/debug_notebooks/data/zheng/zheng_test.h5ad",
                "class_names": "/vevo/gits/mosaicfm/debug_notebooks/data/zheng/zheng-str_label.npy",
            },
            "Segerstolpe": {
                "train": "/vevo/gits/mosaicfm/debug_notebooks/data/Segerstolpe/Segerstolpe_train.h5ad",
                "test": "/vevo/gits/mosaicfm/debug_notebooks/data/Segerstolpe/Segerstolpe_test.h5ad",
                "class_names": "/vevo/gits/mosaicfm/debug_notebooks/data/Segerstolpe/Segerstolpe-str_label.npy",
            },
        }

    def fit_end(self, state: State, logger: Logger):

        # cell classification both for zheng and Segerstolpe datasets
        for dataset in ["zheng", "Segerstolpe"]:
            self.cell_classfication(dataset, logger)

    def cell_classfication(self, dataset: str, logger: Logger):
        # step 1: load data train, test
        class_idx_to_name = np.load(self.dataset_registry[dataset]["class_names"])
        adata_train, gene_ids_train, labels_train, _ = (
            self.prepare_cell_annotation_data(
                self.dataset_registry[dataset]["train"],
                class_idx_to_name,
            )
        )
        adata_test, gene_ids_test, labels_test, _ = self.prepare_cell_annotation_data(
            self.dataset_registry[dataset]["test"],
            class_idx_to_name,
        )

        # step 2: extract mosaicfm embeddings

        cell_embeddings_train = get_batch_embeddings(
            adata=adata_train,
            model=self.model.model,
            vocab=self.vocab,
            gene_ids=gene_ids_train,
            model_cfg=self.model_config,
            collator_cfg=self.collator_config,
            batch_size=256,
            max_length=2048,
            return_gene_embeddings=False,
        )
        cell_embeddings_test = get_batch_embeddings(
            adata=adata_test,
            model=self.model.model,
            vocab=self.vocab,
            gene_ids=gene_ids_test,
            model_cfg=self.model_config,
            collator_cfg=self.collator_config,
            batch_size=256,
            max_length=2048,
            return_gene_embeddings=False,
        )

        # step 3: train classifier
        clf = LogisticRegression(
            max_iter=5000,
            solver="lbfgs",
            multi_class="multinomial",
            random_state=42,
        )
        clf.fit(cell_embeddings_train, labels_train)

        # step 4: calculate and log metrics
        from sklearn.metrics import f1_score

        labels_pred = clf.predict(cell_embeddings_test)
        f1 = f1_score(labels_test, labels_pred, average="macro")

        logger.log_metrics({f"macro_f1_{dataset}": f1})

    def prepare_cell_annotation_data(
        self,
        data_path: str,
        class_idx_to_name: np.ndarray,
    ):

        assert (
            "zheng" in data_path or "Segerstolpe" in data_path
        ), "We currently only supprt Zheng and Segerstolpe datasets!"

        vocab = self.vocab
        adata = sc.read_h5ad(data_path)

        gene_name_key = "gene_symbols"
        gene_col = "gene_id"
        cell_type_key = "cell_type_label"

        adata.var[gene_col] = adata.var[gene_name_key].apply(
            lambda x: self.gene_to_id.get(x, "na"),
        )

        # filter the cell with NaN values in the cell_type_key
        adata = adata[~adata.obs[cell_type_key].isna(), :]

        adata.var["id_in_vocab"] = [vocab.get(gene, -1) for gene in adata.var[gene_col]]
        gene_ids_in_vocab = np.array(adata.var["id_in_vocab"])
        print(
            f"match {np.sum(gene_ids_in_vocab >= 0)}/{len(gene_ids_in_vocab)} genes "
            f"in vocabulary of size {len(vocab)}.",
        )
        adata = adata[:, adata.var["id_in_vocab"] >= 0]
        vocab.default_index = vocab["<pad>"]
        genes = adata.var[gene_col].tolist()
        gene_ids = np.array([vocab[gene] for gene in genes], dtype=int)

        # Extract labels from the AnnData object
        labels = adata.obs[cell_type_key].values
        unique_labels = np.unique(np.array(labels[~np.isnan(np.array(labels))]))

        # Convert labels to numeric if they are not already
        if labels.dtype.kind in "OU":  # Object or Unicode, meaning strings
            label_names = labels
            label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
            labels = np.array([label_to_idx[label] for label in labels])
        else:
            labels = labels.astype(np.int64)
            label_names = np.array([class_idx_to_name[label] for label in labels])

        adata = adata.copy()
        adata.X = adata.X.todense()

        return adata, gene_ids, labels, label_names
