# Copyright (C) Vevo Therapeutics 2024-2025. All rights reserved.
import logging
from typing import Mapping, Optional, Tuple

import torch
import torch.nn.functional as F
from composer.models import ComposerModel
from composer.utils import dist
from llmfoundry.layers_registry import param_init_fns
from omegaconf import DictConfig
from torch import Tensor, nn

from mosaicfm.loss import MaskedMseMetric, MaskedSpearmanMetric, masked_mse_loss
from mosaicfm.model.blocks import (
    CategoryValueEncoder,
    ChemEncoder,
    ContinuousValueEncoder,
    GeneEncoder,
    SCGPTBlock,
    SCGPTEncoder,
    SkipBlock,
    gene_encoder_defaults,
    init_config_defaults,
)

log = logging.getLogger(__name__)


class MosaicFM(nn.Module):
    def __init__(
        self,
        model_config: DictConfig,
        collator_config: DictConfig,
        device: Optional[str] = None,
    ):
        super().__init__()
        self.model_type = "Transformer"
        self.device = device
        self.vocab_size = model_config.vocab_size
        self.n_layers = model_config.n_layers
        self.n_heads = model_config.n_heads
        self.d_model = model_config.d_model
        self.expansion_ratio = model_config.expansion_ratio
        self.norm_scheme = model_config.get("norm_scheme", "pre")
        self.transformer_activation = model_config.get("transformer_activation", "gelu")
        self.use_chem_token = collator_config.get("use_chem_token", False)
        assert (
            not self.use_chem_token or "chemical_encoder" in model_config
        ), "If use_chem_token is set to True, chemical_encoder submodule needs to be specified!"
        assert (
            "chemical_encoder" not in model_config or self.use_chem_token
        ), "If chemical_encoder submodule is specified, use_chem_token needs to be set to True!"

        self.init_device = model_config.get("init_device", "cpu")
        if self.init_device == "mixed":
            if dist.get_local_rank() == 0:
                self.init_device = "cpu"
            else:
                self.init_device = "meta"
        self.cell_emb_style = model_config.get("cell_emb_style", "cls")
        self.pad_token_id = collator_config.pad_token_id
        self.pad_value = collator_config.pad_value
        self.n_input_bins = collator_config.num_bins
        self.attn_config = model_config.get("attn_config", None)
        self.norm_config = model_config.get("norm_config", None)
        self.init_config = model_config.get("init_config", None)
        self.gene_encoder_config = model_config.get("gene_encoder", None)
        if self.init_config is None:
            self.init_config = init_config_defaults
        if self.gene_encoder_config is None:
            self.gene_encoder_config = gene_encoder_defaults
        if self.cell_emb_style not in ["cls", "avg-pool", "w-pool"]:
            raise ValueError(f"Unknown cell_emb_style: {self.cell_emb_style}")

        self.gene_encoder = GeneEncoder(
            self.vocab_size,
            self.d_model,
            activation=self.gene_encoder_config.get("activation", "gelu"),
            padding_idx=self.pad_token_id,
            use_norm=self.gene_encoder_config["use_norm"],
            gene_encoder_cfg=self.gene_encoder_config,
        )

        expression_encoder_config = model_config.expression_encoder
        self.input_emb_style = expression_encoder_config.get(
            "input_emb_style",
            "continuous",
        )
        if self.input_emb_style not in ["category", "continuous"]:
            raise ValueError(
                f"input_emb_style should be one of category or continuous"
                f"got {self.input_emb_style}",
            )
        if self.input_emb_style == "continuous":
            self.expression_encoder = ContinuousValueEncoder(
                d_model=self.d_model,
                dropout=expression_encoder_config.get("dropout", 0.1),
                max_value=expression_encoder_config.get("max_value", 512),
                activation=expression_encoder_config.get("activation", "relu"),
                use_norm=expression_encoder_config.get("use_norm", False),
            )
        elif self.input_emb_style == "category":
            assert self.n_input_bins > 0
            self.expression_encoder = CategoryValueEncoder(
                self.n_input_bins,
                self.d_model,
                padding_idx=self.pad_value,
                use_norm=False,
            )
        else:
            raise ValueError(f"Unknown input_emb_style: {self.input_emb_style}")

        if self.use_chem_token:
            chem_encoder_config = model_config.chemical_encoder
            self.chem_encoder = ChemEncoder(
                drug_fps_path=chem_encoder_config.get("drug_fps_path"),
                d_out=self.d_model,
                padding_idx=chem_encoder_config.get("padding_idx", 0),
                activation=chem_encoder_config.get("activation", "leaky_relu"),
                freeze=chem_encoder_config.get("freeze", False),
            )

        encoder_layers = SCGPTBlock(
            d_model=self.d_model,
            n_heads=self.n_heads,
            expansion_ratio=self.expansion_ratio,
            attn_config=self.attn_config,
            norm_config=self.norm_config,
            activation=self.transformer_activation,
            device=self.device,
            norm_scheme=self.norm_scheme,
            use_glu=model_config.get("use_glu", False),
        )
        self.transformer_encoder = SCGPTEncoder(
            encoder_layers,
            self.n_layers,
            use_norm=self.norm_scheme == "pre",
            norm_config=self.norm_config,
            attn_config=self.attn_config,
        )

        self.binary_decoder = nn.Sequential(
            SkipBlock(2*self.d_model),
            SkipBlock(2*self.d_model),
            nn.Linear(2*self.d_model, 1, bias=True),
        )


        if self.init_device != "meta":
            log.info(
                'MosaicML recommends using config.init_device="meta" with Composer + FSDP for faster initialization.',
            )
            self.apply(self.param_init_fn)

    def param_init_fn(self, module: nn.Module):
        # skip initialization for modules that has skip_init=True
        if hasattr(module, "skip_init") and module.skip_init:
            log.info(f"Skipping re-initializing for {module._get_name()}")
            return
        init_fn_name = self.init_config["name"]
        param_init_fns.get(init_fn_name)(
            module=module,
            n_layers=self.n_layers,
            d_model=self.d_model,
            **self.init_config,
        )

    def forward(
        self,
        *args,
        **kwargs,
    ) -> Mapping[str, Tensor]:
        

        return self.forward_train(*args, **kwargs)


    def forward_train(
        self,
        in_gene_ids: Tensor,
        in_exprs: Tensor,
        high_exp_gene_ids: Tensor,
        high_exp_exprs: Tensor,
        non_exp_gene_ids: Tensor,
        non_exp_exprs: Tensor,
        rand_gene_ids: Tensor,
        rand_exprs: Tensor,
        drug_ids: Optional[Tensor] = None,
    ) -> Mapping[str, Tensor]:
        
        """
        Args:
            in_gene_ids (:obj:`Tensor`): token ids of the input genes, shape
                [batch_size, seq_len]
            in_exprs (:obj:`Tensor`): token values of the input genes, shape
                [batch_size, seq_len]
            high_exp_gene_ids (:obj:`Tensor`): token ids of the high expression genes,
                shape [batch_size, seq_len]
            high_exp_exprs (:obj:`Tensor`): token values of the high expression genes,
                shape [batch_size, seq_len]
            non_exp_gene_ids (:obj:`Tensor`): token ids of the non-expression genes,
                shape [batch_size, seq_len]
            non_exp_exprs (:obj:`Tensor`): token values of the non-expression genes,
                shape [batch_size, seq_len]
            rand_gene_ids (:obj:`Tensor`): token ids of the random genes,
                shape [batch_size, seq_len]
            rand_exprs (:obj:`Tensor`): token values of the random genes,
                shape [batch_size, seq_len]
            drug_ids (:obj:`Tensor`): drug ids corresponding to chem_encoder embedding layer, shape
                [batch_size]

        Returns:
            :obj:`Mapping[str, Tensor]`: output dictionary containing predictions and
            cell embedding.
        """

        output = {}
        len_rand_genes = rand_gene_ids.shape[1]

        # 1) feed input genes and expressions to the transformer
        token_gene_embs = self.gene_encoder(in_gene_ids)  # (batch, in_seq_len, embsize)
        token_values = self.expression_encoder(in_exprs)  # (batch, in_seq_len, embsize)
        total_embs = token_gene_embs + token_values  # (batch, in_seq_len, embsize)

        if self.use_chem_token:
            # calculate chemical embedding and put it in its correct place (after <cls>)
            drug_embs = self.chem_encoder(drug_ids)  # (batch, embsize)
            total_embs[:, 1, :] = drug_embs  # (batch, in_seq_len, embsize)



        transformer_output = self.transformer_encoder(
            pcpt_total_embs=total_embs,
            gen_total_embs=None,
            pcpt_key_padding_mask=None,
            gen_key_padding_mask=None,
        ) # (batch, seq_len, embsize)


        # 2) get CLS embedding (cell embeddings) from the transformer output

        cell_emb = transformer_output[:, 0, :]  # (batch, embsize)
        output["cell_emb"] = cell_emb

        # 3) create output sequence (values will be GT) and feed its gene ids to gene encoder
        out_gene_ids = torch.cat((high_exp_gene_ids, non_exp_gene_ids, rand_gene_ids), dim=1)
        out_exprs = torch.cat((high_exp_exprs, non_exp_exprs, rand_exprs), dim=1)

        output["out_exprs"] = out_exprs # (batch, out_seq_len)
        output["rand_exprs"] = rand_exprs  # (batch, len_rand_genes)
        
        out_gene_embs = self.gene_encoder(out_gene_ids)  # (batch, out_seq_len, embsize)

        # 4) append CLS embedding to each token of output sequence and feed it to decoder to predict expressions values of output genes

        out_token_embs = torch.cat((cell_emb.unsqueeze(1).repeat(1, out_gene_ids.shape[1], 1), out_gene_embs), dim=-1)  # (batch, seq_len, 2*embsize)

        decoder_output = self.binary_decoder(out_token_embs)
        output["out_preds"] = decoder_output.squeeze(-1)
        output["rand_preds"] = decoder_output[:, -len_rand_genes:, :].squeeze(-1)  # (batch, len_rand_genes)

        assert output["rand_exprs"].shape[1] == output["rand_preds"].shape[1], \
            f"rand_exprs shape {output['rand_exprs'].shape} does not match rand_preds shape {output['rand_preds'].shape}"

        return output




class ComposerMosaicFM(ComposerModel):
    def __init__(self, model_config, collator_config, device=None):
        super().__init__()
        self.criterion = masked_mse_loss
        self.pad_token_id = collator_config.pad_token_id

        self.model = MosaicFM(
            model_config=model_config,
            collator_config=collator_config,
            device=device,
        )
        self.n_active_params = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )
        self.train_metrics = {
            "MSE_gene": MaskedMseMetric(name="MSE_gene"),
            "MSE_cell": MaskedMseMetric(name="MSE_cell"),
        }
        self.standard_scale_outputs = model_config.get("standard_scale_outputs", False)
        self.collator_config = collator_config
        self.model_config = model_config

        self.val_metrics = {
            "MSE_gene": MaskedMseMetric(name="MSE_gene"),
            "MSE_cell": MaskedMseMetric(name="MSE_cell"),
            "Spearman_gene": MaskedSpearmanMetric(name="Spearman_gene"),
            "Spearman_cell": MaskedSpearmanMetric(name="Spearman_cell"),
        }

    def forward(self, batch):  # batch is the output of the dataloader
        # specify how batches are passed through the model


        in_gene_ids = batch["in_gene_ids"]
        in_exprs = batch["in_exprs"]
        high_exp_gene_ids = batch["high_exp_gene_ids"]
        high_exp_exprs = batch["high_exp_exprs"]
        non_exp_gene_ids = batch["non_exp_gene_ids"]
        non_exp_exprs = batch["non_exp_exprs"]  
        rand_gene_ids = batch["rand_gene_ids"]
        rand_exprs = batch["rand_exprs"]

        drug_ids = (
            batch["drug_ids"] if "drug_ids" in batch else None
        )  # drug_ids is None if use_chem_token is set to False


        output_dict = self.model(
            in_gene_ids=in_gene_ids,
            in_exprs=in_exprs,
            high_exp_gene_ids=high_exp_gene_ids,
            high_exp_exprs=high_exp_exprs,
            non_exp_gene_ids=non_exp_gene_ids,
            non_exp_exprs=non_exp_exprs,
            rand_gene_ids=rand_gene_ids,
            rand_exprs=rand_exprs,
            drug_ids=drug_ids,
        )

        return output_dict

    def eval_forward(self, batch, outputs: Optional = None):
        if outputs:
            return outputs

        self.model.zero_grad(set_to_none=True)

        return outputs if outputs is not None else self.forward(batch)



    def loss(self, outputs, batch):

        #1) calculate loss output genes
        out_exprs = outputs["out_exprs"]  # GT = (batch, out_seq_len)
        out_preds = outputs["out_preds"]  # preds = (batch, out_seq_len)

        positions_to_match = torch.ones_like(out_exprs, dtype=torch.bool)

        loss_gene = self.criterion(out_preds, out_exprs, positions_to_match)


        #2) calculate loss cell which amplifies loss random genes across the whole batch
        rand_exprs = outputs["rand_exprs"]  # (batch, len_rand_genes)
        rand_preds = outputs["rand_preds"]  # (batch, len_rand_genes)

        positions_to_match = torch.ones_like(rand_exprs, dtype=torch.bool)

        loss_cell = self.criterion(rand_preds, rand_exprs, positions_to_match)  



        #3) combine cell and gene losses

        loss = self.model_config.get("loss_wg", 1.0) * loss_gene + self.model_config.get("loss_wc", 0.0) * loss_cell
        return loss


    def update_metric(self, batch, outputs, metric):

        if "cell" in metric.name:
            preds = outputs["rand_preds"]
            target = outputs["rand_exprs"] 
            mask = ~batch["rand_gene_ids"].eq(self.pad_token_id)
        elif "gene" in metric.name:
            preds = outputs["out_preds"]
            target = outputs["out_exprs"]
        else:
            raise ValueError(f"metric {metric.name} not recognized!")

        mask = torch.ones_like(target, dtype=torch.bool) 


        metric.update(preds=preds, target=target, mask=mask)

    def get_metrics(self, is_train=False):
        # defines which metrics to use in each phase of training
        metric_dict = self.train_metrics if is_train else self.val_metrics
        return metric_dict

    def flops_per_batch(self, batch: Mapping) -> int:
        # specify how to compute the number of FLOPs for a batch
        # This assumes non cell-conditioned generation (single forward pass)
        bs = batch["in_gene_ids"].shape[0]
        msl = batch["in_gene_ids"].shape[1] # Assumes no-padding (as an approximation)
        params = self.n_active_params
        params_flops_per_token = 2 * params
        params_flops_per_seq = params_flops_per_token * msl
        attn_flops_per_seq = (
            self.model.n_layers * 2 * 2 * (self.model.d_model * (msl**2))
        )
        return (params_flops_per_seq + attn_flops_per_seq) * 3 * bs

    def scale_outputs(self, x: torch.Tensor) -> torch.Tensor:
        min_value = 1
        max_value = self.collator_config.num_bins - 1
        normalized_value = (x - min_value) / (max_value - min_value)
        # Scale to -1..1
        return 2 * normalized_value - 1

