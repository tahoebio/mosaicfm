# Copyright (C) Vevo Therapeutics 2025. All rights reserved.
import os

import numpy as np
import pandas as pd
import scanpy as sc
from composer import State
from composer.core.callback import Callback
from composer.loggers import Logger
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from torch.distributed.fsdp.fully_sharded_data_parallel import \
    FullyShardedDataParallel as FSDP

from mosaicfm.model import ComposerSCGPTModel
from mosaicfm.tokenizer import GeneVocab
from mosaicfm.utils import download_file_from_s3_url

class MarginalEssentiality(Callback):
    def __init__(
        self,
        cfg,
        model: ComposerSCGPTModel,
        vocab: GeneVocab,
        model_config,
        collator_config,
        run_name,
    ):

        super().__init__()
        model.eval()
        self.model = model
        self.vocab = vocab
        model_config["precision"] = "amp_bf16"
        self.model_config = model_config
        self.collator_config = collator_config
        self.run_name = run_name
        self.task_cfg = cfg
        self.batch_size = self.task_cfg.get("batch_size", 32)
        self.seq_len = self.task_cfg.get("seq_len", 8192)
        self.rf_jobs = self.task_cfg.get("rf_jobs", 8)

    def fit_end(self, state: State, logger: Logger):

        # download task data from S3
        local_adata_path = os.path.join(self.task_cfg["local_dir"], "ccle.h5ad")
        local_label_path = os.path.join(self.task_cfg["local_dir"], "labels.csv")
        download_file_from_s3_url(s3_url=os.path.join(self.task_cfg["remote_dir"], "ccle.h5ad"), local_file_path=local_adata_path)
        download_file_from_s3_url(s3_url=os.path.join(self.task_cfg["remote_dir"], "labels.csv"), local_file_path=local_label_path)

        # load and process AnnData of CCLE counts
        vocab = self.vocab
        adata = sc.read_h5ad(local_adata_path)
        adata.var["id_in_vocab"] = [vocab[gene] if gene in vocab else -1 for gene in adata.var["feature_id"]]
        adata = adata[:, adata.var["id_in_vocab"] >= 0]
        gene_ids_in_vocab = np.array(adata.var["id_in_vocab"])
        genes = adata.var["feature_id"].tolist()
        gene_ids = np.array([vocab[gene] for gene in genes], dtype=int)
        print(f"matched {np.sum(gene_ids_in_vocab >= 0)}/{len(gene_ids_in_vocab)} genes in vocabulary of size {len(vocab)}")

        # get gene embeddings
        with FSDP.summon_full_params(self.model.model):
            _, gene_embeddings = get_batch_embeddings(
                adata=adata,
                model=self.model.model.module,
                vocab=self.vocab,
                gene_ids=gene_ids,
                model_cfg=self.model_config,
                collator_cfg=self.collator_config,
                batch_size=self.batch_size,
                max_length=self.seq_len,
                return_gene_embeddings=True
            )

        # load task DataFrame
        task_df = pd.read_csv(local_label_path)
        genes = task_df["gene_id"].to_numpy()
        labels = task_df["essential"].to_numpy()

        # get mean embeddings for each gene
        gene2idx = vocab.get_stoi()
        gene_names = list(gene2idx.keys())
        mean_embs = np.zeros((len(genes), gene_embeddings.shape[1]))
        for i, g in enumerate(genes):
            mean_embs[i] = gene_embeddings[np.where(gene_names == g)[0][0]]

        # split into training and testing sets
        emb_train, emb_test, labels_train, labels_test = train_test_split(
            mean_embs,
            labels,
            test_size=0.2,
            random_state=42
        )

        # train classifer and report auROC on test set
        rf = RandomForestClassifier(n_jobs=self.rf_jobs)
        rf.fit(emb_train, labels_train)
        test_probas = rf.predict_proba(emb_test)
        auroc = float(roc_auc_score(labels_test, test_probas[:, 1]))
        logger.log_metrics({"marginal gene essentiality auROC": auroc})