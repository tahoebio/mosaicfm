# Copyright (C) Vevo Therapeutics 2025. All rights reserved.
import composer
import torch
from composer.core import State
from llmfoundry.registry import algorithms
from torch.optim.swa_utils import SWALR, AveragedModel


class SWA_modified(composer.algorithms.SWA):
    def _initialize_swa(self, state: State) -> None:
        """Override _initialize_swa to customize how the model's state_dict is
        loaded."""
        if self.schedule_swa_lr:
            self.swa_lr = self._get_last_lr(state.schedulers)

            if len(state.optimizers) != 1:
                raise RuntimeError("SWA supports only one optimizer")

            self.swa_scheduler = SWALR(
                state.optimizers[0],
                swa_lr=self.swa_lr,
                anneal_epochs=self.anneal_steps,
                anneal_strategy=self.anneal_strategy,  # type: ignore
            )

        # If swa_model is already initialized, just reset the state_dict
        if self.swa_model is None:
            # If swa_model is not initialized, initialize it
            self.swa_model = AveragedModel(state.model, device=torch.device("cpu"))
        else:
            # If swa_model is already initialized, modify state_dict loading
            current_state_dict = state.model.state_dict()
            # Add the 'module.' prefix to the current model's state_dict keys to match AveragedModel
            current_state_dict = {
                f"module.{k}": v for k, v in current_state_dict.items()
            }

            # Now load the modified state_dict (with the 'module.' prefix) into swa_model
            self.swa_model.load_state_dict(current_state_dict, strict=False)


algorithms.register("swa", func=SWA_modified)
