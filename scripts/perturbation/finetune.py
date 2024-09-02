# Copyright (C) Vevo Therapeutics 2024. All rights reserved.
import copy
import logging
import os
import sys
from typing import Any, Dict, List, Optional, Union

import composer
from composer.core.callback import Callback
from composer.utils import dist, get_device, reproducibility
from llmfoundry.utils.builders import (
    build_algorithm,
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
from perturb_algorithm import SetFindUnusedParameters
from perturb_callback import PerturbationCallback

from mosaicfm.data import build_perturbation_dataloader
from mosaicfm.model import ComposerSCGPTPerturbationModel
from mosaicfm.utils import download_file_from_s3_url, load_mean_ctrl

log = logging.getLogger(__name__)
os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"


def main(cfg: DictConfig) -> composer.Trainer:

    cfg = update_batch_size_info(cfg)
    # Create copy of config for logging
    logged_cfg: DictConfig = copy.deepcopy(cfg)
    # Get global and device batch size information from distributed/single node setting
    logged_cfg.update(cfg, merge=True)

    # Set seed first
    seed: int = pop_config(cfg, "seed", must_exist=True)
    reproducibility.seed_all(seed)

    # Initialize pytorch distributed training process groups
    dist_timeout: Union[int, float] = pop_config(
        cfg,
        "dist_timeout",
        must_exist=False,
        default_value=600.0,
    )
    dist.initialize_dist(get_device(None), timeout=dist_timeout)

    # Mandatory hyperparameters for training
    device_train_batch_size: int = pop_config(
        cfg,
        "device_train_batch_size",
        must_exist=True,
    )
    device_test_batch_size: int = pop_config(
        cfg,
        "device_test_batch_size",
        must_exist=True,
    )
    max_duration: str = pop_config(cfg, "max_duration", must_exist=True)
    precision: str = pop_config(cfg, "precision", must_exist=True)

    # Optional parameters will be set to default values if not specified.
    run_name: str = pop_config(
        cfg,
        "run_name",
        must_exist=False,
        default_value="thetaf-test",
    )
    logged_cfg.update({"run_name": run_name})

    save_folder: Optional[str] = pop_config(
        cfg,
        "save_folder",
        must_exist=False,
        default_value=f"s3://vevo-ml-datasets/vevo-scgpt/models/{run_name}",
    )
    save_overwrite: bool = pop_config(
        cfg,
        "save_overwrite",
        must_exist=False,
        default_value=False,
    )
    save_weights_only: bool = pop_config(
        cfg,
        "save_weights_only",
        must_exist=False,
        default_value=False,
    )

    # Enable autoresume from model checkpoints if possible
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

    autoresume: bool = pop_config(
        cfg,
        "autoresume",
        must_exist=False,
        default_value=autoresume_default,
    )

    log.info("Downloading pretrained model ...")

    # Pop necessary configs
    model_cfg: DictConfig = pop_config(cfg, "model", must_exist=True, convert=True)

    # model_cfg_path: str = model_cfg.get("cfg_path")
    model_cfg_path: DictConfig = pop_config(model_cfg, "cfg_path", must_exist=True)

    if dist.get_local_rank() == 0:
        download_file_from_s3_url(
            s3_url=model_cfg_path.get("remote"),
            local_file_path=model_cfg_path.get("local"),
        )
    with dist.local_rank_zero_download_and_wait(model_cfg_path.get("local")):
        dist.barrier()

    model_config = om.load(model_cfg_path.get("local"))

    # model_collator_cfg_path = model_cfg.get("collator_cfg_path")
    model_collator_cfg_path: DictConfig = pop_config(
        model_cfg,
        "collator_cfg_path",
        must_exist=True,
    )

    if dist.get_local_rank() == 0:
        download_file_from_s3_url(
            s3_url=model_collator_cfg_path.get("remote"),
            local_file_path=model_collator_cfg_path.get("local"),
        )
    with dist.local_rank_zero_download_and_wait(model_collator_cfg_path.get("local")):
        dist.barrier()
    model_collator_cfg = om.load(model_collator_cfg_path.get("local"))

    freeze = model_cfg.get(
        "freeze",
        False,
    )  # if the transformer (SCGPT model) is freezed or not!
    pretrained = model_cfg.get(
        "pretrained",
        True,
    )  # if the transformer weights are initialized with pretrained model or random!
    model_file = model_cfg.get("checkpoint_path") if pretrained else None

    # Load datasets and build loaders
    train_loader_cfg: DictConfig = pop_config(cfg, "train_loader", must_exist=True)
    valid_loader_cfg: DictConfig = pop_config(cfg, "valid_loader", must_exist=True)
    test_loader_cfg: DictConfig = pop_config(cfg, "test_loader", must_exist=True)

    train_loader = build_perturbation_dataloader(
        loader_cfg=train_loader_cfg,
        device_batch_size=device_train_batch_size,
        isTrain=True,
    )

    valid_loader = build_perturbation_dataloader(
        loader_cfg=valid_loader_cfg,
        device_batch_size=device_test_batch_size,
        isTrain=False,
    )

    test_loader = build_perturbation_dataloader(
        loader_cfg=test_loader_cfg,
        device_batch_size=device_test_batch_size,
        isTrain=False,
    )

    log.info(f"Training set number of samples: {len(train_loader)}")
    log.info(f"Validation set number of samples: {len(valid_loader)}")
    log.info(f"Test set number of samples: {len(test_loader)}")

    logged_cfg.update(
        {
            "train_dataset_size": len(train_loader),
            "valid_dataset_size": len(valid_loader),
            "test_dataset_size": len(test_loader),
        },
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

    algorithm_configs: Optional[DictConfig] = pop_config(
        cfg,
        "algorithms",
        must_exist=False,
        default_value=None,
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

    path_mean_ctrl: DictConfig = pop_config(
        valid_loader_cfg,
        "path_mean_ctrl",
        must_exist=True,
    )

    if dist.get_local_rank() == 0:
        download_file_from_s3_url(
            s3_url=path_mean_ctrl.get("remote"),
            local_file_path=path_mean_ctrl.get("local"),
        )
    with dist.local_rank_zero_download_and_wait(path_mean_ctrl.get("local")):
        dist.barrier()

    mean_ctrl = load_mean_ctrl(path_mean_ctrl.get("local"))

    pert_callback = PerturbationCallback(mean_ctrl)
    callbacks.append(pert_callback)

    # Algorithms
    algorithms = (
        [
            build_algorithm(str(name), algorithm_cfg)
            for name, algorithm_cfg in algorithm_configs.items()
        ]
        if algorithm_configs
        else None
    )

    algorithms.append(SetFindUnusedParameters())

    # Load model
    model = ComposerSCGPTPerturbationModel(
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
    scheduler_config: Dict[str, Any] = pop_config(
        cfg,
        "scheduler",
        must_exist=True,
        convert=True,
    )
    scheduler_name: str = scheduler_config.pop("name")
    scheduler = build_scheduler(scheduler_name, scheduler_config)

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
        algorithms=algorithms,
        train_dataloader=train_loader,
        eval_dataloader=valid_loader,
        load_strict_model_weights=False,
        precision=precision,
        load_path=model_file,
        load_weights_only=True,
        load_ignore_keys=["state/model/pert_encoder*", "state/model/pert_decoder*"],
        save_folder=save_folder,
        max_duration=max_duration,
        autoresume=autoresume,
        dist_timeout=dist_timeout,
    )

    # Train
    log.info("Starting training...")
    trainer.fit()
    log.info("Training finished.")

    # Run evaluation on the test_loader using the trainer's built-in eval method
    trainer.eval(
        eval_dataloader=test_loader,
        # callbacks= pert_callback,  # Use the callback to collect and log metrics
        subset_num_batches=None,  # Evaluate on the entire test set
    )

    print("Test set evaluation completed.")
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
