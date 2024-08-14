# Copyright (C) Vevo Therapeutics 2024. All rights reserved.
import numpy as np
from composer.core.callback import Callback

from mosaicfm.utils import calc_pearson_metrics


# Define your custom callback (ensure it's defined somewhere in your codebase)
class PerturbationCallback(Callback):
    def __init__(self, mean_ctrl, non_zero_genes=False):

        super().__init__()
        self.non_zero_genes = non_zero_genes
        self.preds = []
        self.targets = []
        self.conditions = []
        self.mean_ctrl = mean_ctrl  # (n_genes,)

    def eval_start(self, state, logger):
        # Clear predictions and labels at the start of evaluation
        self.preds.clear()
        self.targets.clear()
        self.conditions.clear()

        print("Collecting predictions started.")

    def eval_batch_end(self, state, logger):

        # Collect predictions and true labels from the batch
        model_output = state.outputs
        batch = state.batch

        # Assuming model_output and batch contain the necessary data
        preds = model_output["predicted_expr_perturbed"].detach().cpu().numpy()
        targets = batch["expressions_perturbed"].detach().cpu().numpy()
        conditions = batch["perturb_names"]

        # mean_ctrl = batch['mean_ctrl'].detach().cpu().numpy()

        self.preds.append(preds)
        self.targets.append(targets)
        self.conditions.append(conditions)

    def eval_end(self, state, logger):

        # Concatenate all predictions and labels
        preds = np.concatenate(self.preds, axis=0)
        targets = np.concatenate(self.targets, axis=0)
        conditions = np.concatenate(self.conditions, axis=0)

        print("Evaluation ended. Total predictions collected:", len(preds))

        # Compute Pearson metrics
        metrics = calc_pearson_metrics(preds, targets, conditions, self.mean_ctrl)

        # Log metrics
        logger.log_metrics(metrics)
