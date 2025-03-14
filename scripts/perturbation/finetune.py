# Copyright (C) Vevo Therapeutics 2024. All rights reserved.
import copy
import logging
import sys
import json
import gc
from typing import Any, Dict, List, Optional

import composer
import torch
import numpy as np
import torch.utils
import torch.utils.data
import scanpy as sc
from composer.core.callback import Callback
from composer.utils import reproducibility
from llmfoundry.utils.builders import (
    build_callback,
    build_logger,
    build_optimizer,
    build_scheduler,
)
from llmfoundry.utils.config_utils import (
    pop_config,
    update_batch_size_info,
)
from omegaconf import DictConfig
from omegaconf import OmegaConf as om
from torch.utils.data import random_split

from mosaicfm.model import ComposerSCGPTModel
from mosaicfm.tokenizer import GeneVocab
from mosaicfm.data import CountDataset, DataCollator

from composer.optim import ConstantScheduler

log = logging.getLogger(__name__)


def main(cfg: DictConfig) -> composer.Trainer:

    # create copy of config for logging
    cfg = update_batch_size_info(cfg)
    logged_cfg: DictConfig = copy.deepcopy(cfg)
    logged_cfg.update(cfg, merge=True)

    # set seed first
    seed: int = pop_config(cfg, "seed", must_exist=True)
    reproducibility.seed_all(seed)

    # mandatory hyperparameters for training
    device_train_batch_size: int = pop_config(cfg, "device_train_batch_size", must_exist=True)
    device_test_batch_size: int = pop_config(cfg, "device_test_batch_size", must_exist=True)
    max_duration: str = pop_config(cfg, "max_duration", must_exist=True)
    precision: str = pop_config(cfg, "precision", must_exist=True)

    # optional parameters will be set to default values if not specified
    run_name: str = pop_config(cfg, "run_name", must_exist=False, default_value="thetaf-test")
    logged_cfg.update({"run_name": run_name})
    save_folder: Optional[str] = pop_config(cfg, "save_folder", must_exist=False, default_value=f"s3://vevo-ml-datasets/vevo-scgpt/models/{run_name}")
    save_overwrite: bool = pop_config(cfg, "save_overwrite", must_exist=False, default_value=False)
    save_weights_only: bool = pop_config(cfg, "save_weights_only", must_exist=False, default_value=False)

    # enable autoresume from model checkpoints if possible
    autoresume_default: bool = False
    if (
        logged_cfg.get("run_name", None) is not None
        and save_folder is not None
        and not save_overwrite
        and not save_weights_only
    ):
        autoresume_default = True

    if cfg.get("autoresume") is None and autoresume_default:
        log.info(
            "As run_name, save_folder, and save_latest_filename are set, \
                    changing autoresume default to True...",
        )

    autoresume: bool = pop_config(cfg, "autoresume", must_exist=False, default_value=autoresume_default)

    # pop necessary configs
    model_cfg: Dict[str, Any] = pop_config(cfg, "model", must_exist=True, convert=True)
    model_cfg_path: str = model_cfg.get("cfg_path")
    model_config = om.load(model_cfg_path)
    model_collator_cfg_path = model_cfg.get("collator_cfg_path")
    model_collator_cfg = om.load(model_collator_cfg_path)
    freeze = model_cfg.get("freeze", False)
    pretrained = model_cfg.get("pretrained", True)
    model_file = model_cfg.get("checkpoint_path") if pretrained else None

    # load resistance is futile adata and vocabulary
    data_path = "/vevo/umair/data/rif-adata/adata_clean.h5ad.gz"
    vocab_path = "/vevo/umair/data/scgpt-models/scgpt-70m-1024-fix-norm-apr24-data/vocab.json"
    vocab = GeneVocab.from_file(vocab_path)
    adata = sc.read_h5ad(data_path)
    with open("/vevo/umair/data/rif-adata/gene_info_2024-04-29.json") as f:
        gene_to_ensembl = json.load(f)
        ensembl_to_gene_name = {v: k for k, v in gene_to_ensembl.items()}

    # create dataset
    adata.var["id_in_vocab"] = [vocab[ensembl_to_gene_name.get(gene_id, "<pad>")] for gene_id in adata.var["gene_id"]]
    gene_ids_in_vocab = np.array(adata.var["id_in_vocab"])
    filter_vocab = adata.var["id_in_vocab"] != vocab["<pad>"]
    log.info(f"matched {np.sum(filter_vocab)} / {len(gene_ids_in_vocab)} genes in vocabulary of size {len(vocab)}")
    adata = adata[:, filter_vocab]
    count_matrix = adata.X
    count_matrix = (count_matrix if isinstance(count_matrix, np.ndarray) else count_matrix.A)
    dataset = CountDataset(
        count_matrix,
        gene_ids = np.array(adata.var["id_in_vocab"]),
        cls_token_id=vocab["<cls>"],
        pad_value=model_collator_cfg["pad_value"],
    )

    # split into training and testing
    train_dataset, val_dataset = random_split(dataset, lengths = [0.99, 0.01])

    # build data collator
    collate_fn = DataCollator(
        do_padding=model_collator_cfg.get("do_padding", True),
        pad_token_id=model_collator_cfg.pad_token_id,
        pad_value=model_collator_cfg.pad_value,
        do_mlm=True,  # Enable masking for continual training
        do_binning=model_collator_cfg.get("do_binning", True),
        mlm_probability=model_collator_cfg.mlm_probability,  # Not used
        mask_value=model_collator_cfg.mask_value,
        max_length=1024,
        sampling=model_collator_cfg.sampling,  # Turned on since max-length can be less than the number of genes
        data_style="both",  # Enable generative training
        num_bins=model_collator_cfg.get("num_bins", 51),
        right_binning=model_collator_cfg.get("right_binning", False),
    )

    # build train loader
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=device_train_batch_size,
        collate_fn=collate_fn,
        drop_last=False,
        num_workers=8,
        pin_memory=True,
        prefetch_factor=48,
        persistent_workers=True
    )

    # build val loader
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        shuffle=False,
        batch_size=device_test_batch_size,
        collate_fn=collate_fn,
        drop_last=False,
        num_workers=8,
        pin_memory=True,
        prefetch_factor=48,
        persistent_workers=True
    )

    # Optional logging, evaluation and callback configs
    logger_configs: Optional[DictConfig] = pop_config(
        cfg,
        "loggers",
        must_exist=False,
        default_value=None,
        convert=True,
    )
    callback_configs: Optional[DictConfig] = pop_config(
        cfg,
        "callbacks",
        must_exist=False,
        default_value=None,
        convert=True,
    )

    # Loggers
    loggers = (
        [
            build_logger(str(name), logger_cfg)
            for name, logger_cfg in logger_configs.items()
        ]
        if logger_configs
        else []
    )

    # Callbacks
    callbacks: List[Callback] = (
        [
            build_callback(str(name), callback_cfg, om.to_container(logged_cfg))
            for name, callback_cfg in callback_configs.items()
        ]
        if callback_configs
        else []
    )

    # load model
    model = ComposerSCGPTModel(
        model_config=model_config,
        collator_config=model_collator_cfg,
    )

    # Freeze transformer layers if necessary
    if freeze:
        for (
            param
        ) in model.model.transformer_encoder.parameters():  # model.model is SCGPTModel
            param.requires_grad = False

    # Optimizer
    optimizer_config: Dict[str, Any] = pop_config(
        cfg,
        "optimizer",
        must_exist=True,
        convert=True,
    )
    optimizer_name: str = optimizer_config.pop("name")
    optimizer = build_optimizer(model, optimizer_name, optimizer_config)

    # Scheduler
    # scheduler_config: Dict[str, Any] = pop_config(
    #     cfg,
    #     "scheduler",
    #     must_exist=True,
    #     convert=True,
    # )
    # scheduler_name: str = scheduler_config.pop("name")
    # scheduler = build_scheduler(scheduler_name, scheduler_config)
    scheduler = ConstantScheduler()

    # Generate trainer
    log.info("Building Trainer...")

    trainer = composer.Trainer(
        run_name=run_name,
        seed=seed,
        model=model,
        optimizers=optimizer,
        schedulers=scheduler,
        loggers=loggers,
        callbacks=callbacks,
        train_dataloader=train_loader,
        eval_dataloader=val_loader,
        # load_strict_model_weights=False,
        precision=precision,
        # load_path=model_file,
        # load_weights_only=True,
        # load_ignore_keys=["state/model/pert_encoder*", "state/model/pert_decoder*"],
        max_duration=max_duration,
        autoresume=autoresume,
        save_folder=save_folder,
        save_overwrite=save_overwrite
    )

    # train
    torch.cuda.empty_cache()
    gc.collect()
    log.info("Starting training...")
    trainer.fit()
    log.info("Training finished.")
    return trainer


if __name__ == "__main__":

    yaml_path, args_list = sys.argv[1], sys.argv[2:]
    # Disable resolving environment variables through omegaconf.
    om.clear_resolver("oc.env")
    # Load yaml and cli arguments.
    with open(yaml_path) as f:
        yaml_cfg = om.load(f)
    cli_cfg = om.from_cli(args_list)
    cfg = om.merge(yaml_cfg, cli_cfg)
    om.resolve(cfg)
    assert isinstance(cfg, DictConfig)
    main(cfg)
