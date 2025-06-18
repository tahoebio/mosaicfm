# Copyright (C) Vevo Therapeutics 2024-2025. All rights reserved.
import logging
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from composer.utils import dist
from llmfoundry.layers_registry import attention_classes, norms
from llmfoundry.models.layers.ffn import (
    resolve_ffn_act_fn,
    resolve_ffn_hidden_size,
)
from llmfoundry.models.mpt.modeling_mpt import gen_flash_attn_padding_info
from torch import Tensor, nn
from torch.nn.modules.transformer import _get_clones

from mosaicfm.utils import download_file_from_s3_url

attn_config_defaults: Dict = {
    "attn_type": "grouped_query_attention",
    "attn_pdrop": 0.0,
    "attn_impl": "torch",
    "use_attn_mask": True,
    "qk_ln": False,
    "qk_gn": False,
    "clip_qkv": None,
    "softmax_scale": None,
}

norm_config_defaults: Dict = {
    "norm_type": "low_precision_layernorm",
    "eps": 1e-5,
}

init_config_defaults: Dict = {
    "name": "kaiming_normal_",
    "fan_mode": "fan_in",
    "init_nonlinearity": "relu",
    "init_div_is_residual": True,
    "emb_init_std": None,
    "emb_init_uniform_lim": None,
    "init_std": None,
    "init_gain": 0.0,
}

gene_encoder_defaults: Dict = {
    "use_norm": False,
}

log = logging.getLogger(__name__)


class SCGPTBlock(nn.Module):
    r"""TransformerEncoderLayer is made up of self-attn and feedforward network.
    The class is modified from torch.nn.TransformerEncoderLayer to support the
    FlashAttention.

    Args:
        d_model: the number of expected features in the input (required).
        n_heads: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of intermediate layer, relu or gelu (default=relu).
        layer_norm_eps: the eps value in layer normalization components (default=1e-5).
        batch_first: If ``True``, then the input and output tensors are provided
            as (batch, seq, feature). Default: ``False``.

    Examples::
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        >>> src = torch.rand(10, 32, 512)
        >>> out = encoder_layer(src)

    Alternatively, when ``batch_first`` is ``True``:
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8, batch_first=True)
        >>> src = torch.rand(32, 10, 512)
        >>> out = encoder_layer(src)
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        expansion_ratio: int,
        attn_config: Optional[Dict] = None,
        norm_config: Optional[Dict] = None,
        dropout: Optional[float] = 0.0,
        activation: Optional[str] = "gelu",
        device: Optional[str] = None,
        dtype=None,
        norm_scheme="pre",
        use_glu: bool = False,
        **kwargs: Any,
    ) -> None:
        if attn_config is None:
            attn_config = attn_config_defaults
        if norm_config is None:
            norm_config = norm_config_defaults
        del kwargs  # unused, just to capture any extra args from the config
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        attn_class = attention_classes.get(attn_config["attn_type"])
        self.d_model = d_model
        self.n_heads = n_heads
        self.device = device
        self.self_attn = attn_class(
            d_model=d_model,
            n_heads=n_heads,
            kv_n_heads=attn_config.get("kv_n_heads", n_heads),
            attn_impl=attn_config.get("attn_impl", "triton"),
            device=device,
        )
        # Implementation of Feedforward model
        dim_feedforward = resolve_ffn_hidden_size(d_model, expansion_ratio)
        self.up_proj = nn.Linear(d_model, dim_feedforward, **factory_kwargs)
        self.down_proj = nn.Linear(dim_feedforward, d_model, **factory_kwargs)
        self.use_glu = use_glu
        if self.use_glu:
            self.gate_proj = nn.Linear(d_model, dim_feedforward, **factory_kwargs)

        # Norms
        norm_class = norms.get(norm_config["norm_type"].lower())
        self.norm1 = norm_class(
            d_model,
            device=device,
            eps=norm_config.get("eps", 1e-5),
        )
        self.norm2 = norm_class(
            d_model,
            device=device,
            eps=norm_config.get("eps", 1e-5),
        )
        self.post_sa_dropout = nn.Dropout(dropout)
        self.post_ffn_dropout = nn.Dropout(dropout)

        self.activation = self._get_activation_fn(activation)
        self.norm_scheme = norm_scheme
        if norm_scheme not in ["pre", "post"]:
            raise ValueError("norm_scheme must be either pre or post")

    @staticmethod
    def _get_activation_fn(activation):
        return resolve_ffn_act_fn({"name": activation})

    def forward(
        self,
        x: Tensor,
        attn_bias: Optional[Tensor] = None,
        flash_attn_padding_info: Optional[Tensor] = None,
    ) -> Tensor:
        r"""Pass the input through the encoder layer.

        Args:
            src: the sequence to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """

        if self.norm_scheme == "pre":
            x = x + self._sa_block(
                self.norm1(x),
                attn_bias=attn_bias,
                flash_attn_padding_info=flash_attn_padding_info,
            )
            x = x + self._ff_block(self.norm2(x))
        else:
            x = self.norm1(
                x
                + self._sa_block(
                    x,
                    attn_bias=attn_bias,
                    flash_attn_padding_info=flash_attn_padding_info,
                ),
            )
            x = self.norm2(x + self._ff_block(x))
        return x

    def _sa_block(
        self,
        x: Tensor,
        attn_bias: Optional[Tensor] = None,
        flash_attn_padding_info: Optional[Tensor] = None,
    ) -> Tensor:
        x, _, _ = self.self_attn(
            x,
            attn_bias=attn_bias,
            flash_attn_padding_info=flash_attn_padding_info,
            is_causal=False,
        )
        return self.post_sa_dropout(x)

    def _ff_block(self, x: Tensor) -> Tensor:
        if self.use_glu:
            x = self.down_proj(self.activation(self.gate_proj(x)) * self.up_proj(x))
        else:
            x = self.down_proj(self.activation(self.up_proj(x)))
        return self.post_ffn_dropout(x)


class SCGPTEncoder(nn.Module):
    # takes in the set of different inputs in an mapping
    r"""TransformerEncoder is a stack of N encoder layers. Users can build the
    BERT(https://arxiv.org/abs/1810.04805) model with corresponding parameters.

    Args:
        encoder_layer: an instance of the TransformerEncoderLayer() class (required).
        num_layers: the number of sub-encoder-layers in the encoder (required).
        norm: the layer normalization component (optional).
        enable_nested_tensor: if True, input will automatically convert to nested tensor
            (and convert back on output). This will improve the overall performance of
            TransformerEncoder when padding rate is high. Default: ``True`` (enabled).

    Examples::
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        >>> transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)
        >>> src = torch.rand(10, 32, 512)
        >>> out = transformer_encoder(src)
    """

    def __init__(
        self,
        encoder_layer: SCGPTBlock,
        num_layers: int,
        use_norm: bool = False,
        norm_config: Optional[Dict] = None,
        attn_config: Optional[Dict] = None,
    ):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.use_norm = use_norm

        if attn_config is None:
            attn_config = attn_config_defaults
        self.use_attn_mask = attn_config.get("use_attn_mask", True)
        if self.use_norm:
            if norm_config is None:
                norm_config = norm_config_defaults
            norm_class = norms.get(norm_config["norm_type"].lower())
            self.norm = norm_class(
                encoder_layer.d_model,
                device=encoder_layer.device,
                eps=norm_config.get("eps", 1e-5),
            )

    def forward(
        self,
        pcpt_total_embs: Tensor,
        gen_total_embs: Optional[Tensor] = None,
        pcpt_key_padding_mask: Optional[Tensor] = None,
        gen_key_padding_mask: Optional[Tensor] = None,
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        if gen_total_embs is None:
            pcpt_only = True
            total_embs = pcpt_total_embs
            key_padding_mask = pcpt_key_padding_mask
        else:
            pcpt_only = False
            total_embs = torch.cat([pcpt_total_embs, gen_total_embs], dim=1)
            if pcpt_key_padding_mask is None and gen_key_padding_mask is None:
                key_padding_mask = None
            else:
                if pcpt_key_padding_mask is None:
                    pcpt_key_padding_mask = torch.ones(
                        (pcpt_total_embs.shape[0], pcpt_total_embs.shape[1]),
                        device=pcpt_total_embs.device,
                        dtype=torch.bool,
                    )  # 1 means attention is allowed
                elif gen_key_padding_mask is None:
                    gen_key_padding_mask = torch.ones(
                        (gen_total_embs.shape[0], gen_total_embs.shape[1]),
                        device=gen_total_embs.device,
                        dtype=torch.bool,
                    )  # 1 means attention is allowed
                key_padding_mask = torch.cat(
                    [pcpt_key_padding_mask, gen_key_padding_mask],
                    dim=1,
                )  # (B, S)
        p_len = pcpt_total_embs.shape[1]
        total_len = total_embs.shape[1]
        g_len = total_len - p_len
        flash_attn_padding_info = gen_flash_attn_padding_info(
            bsz=total_embs.shape[0],
            S=total_len,
            past_key_len=0,
            attention_mask=key_padding_mask,
            device=total_embs.device,
        )
        attn_bias = None
        if self.use_attn_mask:
            attention_mask = self._make_mask(p_len, g_len, total_embs.device)
            attn_bias = torch.zeros_like(
                attention_mask,
                dtype=total_embs.dtype,
                device=attention_mask.device,
                requires_grad=False,
            ).masked_fill(
                ~attention_mask,
                torch.finfo(total_embs.dtype).min,
            )  # Matrix with -inf at the place of masked values and 0 elsewhere
            attn_bias = attn_bias.unsqueeze(0).unsqueeze(
                1,
            )  # Broadcastable to (B,H, S_Q, S_K) dimensions

            if (
                key_padding_mask is not None
            ):  # NOTE: handle when key_padding_mask is None
                # Merge the key_padding_mask into attn_bias
                b_size, s_k = key_padding_mask.shape[:2]
                attn_bias = attn_bias.masked_fill(
                    ~key_padding_mask.view((b_size, 1, 1, s_k)),
                    torch.finfo(total_embs.dtype).min,
                )
        for mod in self.layers:
            total_embs = mod(
                total_embs,
                attn_bias=attn_bias,
                flash_attn_padding_info=flash_attn_padding_info,
            )

        if self.use_norm:
            total_embs = self.norm(total_embs)
        if pcpt_only:
            return total_embs
        else:
            pcpt_total_embs = total_embs[:, :p_len, :]
            gen_total_embs = total_embs[:, p_len:, :]
            return pcpt_total_embs, gen_total_embs

    @torch.no_grad()
    @lru_cache(maxsize=1)
    def _make_mask(self, p_len, g_len, device):
        # Mask follows the LLM Foundry convention
        # ie: 0 indicates no-attention, 1 indicates attention is allowed
        total_len = p_len + g_len
        attention_mask = torch.ones(
            (total_len, total_len),
            device=device,
            dtype=torch.bool,
        )  # (pcpt_len+gen_len, pcpt_len+gen_len)

        if g_len > 0:
            # pcpt genes should not see gen genes
            # Equivalent to dense self-attention on pcpt genes
            attention_mask[0:p_len, -g_len:] = False
            # gen genes can see all pcpt genes and themselves, not other gen genes.
            # make the last gen_len by gen_gen to be an identity matrix, attention allowed along the diagonal
            # Equivalent to cross-attention from pcpt genes to gen genes
            attention_mask[-g_len:, -g_len:] = torch.eye(
                g_len,
                device=device,
                dtype=torch.bool,
            )
        return attention_mask


# Various combination strategies for embedding combination
class ElementwiseSumCombination(nn.Module):
    """Combines embeddings with simple element-wise summation."""

    def __init__(self, target_dim: int, num_embedding_types: int):
        super().__init__()
        self.skip_init = True  # No parameters to initialize

    def forward(self, embeddings_list: List[Tensor]) -> Tensor:
        """Simply sum all embeddings element-wise."""
        if len(embeddings_list) == 1:
            return embeddings_list[0]

        return torch.stack(embeddings_list).sum(dim=0)


class LearnedWeightedCombination(nn.Module):
    """Combines embeddings using learned per-dimension weights."""

    def __init__(self, target_dim: int, num_embedding_types: int):
        super().__init__()
        # Initialize with equal weights (1/N for each embedding type)
        self.weights = nn.Parameter(
            torch.ones(target_dim, num_embedding_types) / num_embedding_types,
        )
        self.skip_init = True  # Skip initialization in param_init_fn

    def forward(self, embeddings_list: List[Tensor]) -> Tensor:
        """Apply learned per-dimension weights to combine embeddings."""
        if len(embeddings_list) == 1:
            return embeddings_list[0]

        # Stack embeddings along a new dimension
        # [batch_size, seq_len, embed_dim] → [batch_size, seq_len, num_embeddings, embed_dim]
        stacked_embeddings = torch.stack(embeddings_list, dim=2)

        # Apply per-dimension weighting
        # Einsum notation: b=batch, s=sequence, e=embedding_type, d=dimension
        # [b,s,e,d] x [d,e] → [b,s,d]
        result = torch.einsum("bsed,de->bsd", stacked_embeddings, self.weights)

        return result


class SoftmaxWeightedCombination(nn.Module):
    """Combines embeddings using softmax-normalized weights per dimension."""

    def __init__(self, target_dim: int, num_embedding_types: int):
        super().__init__()
        # Initialize logits for softmax to zeros (equal weighting initially)
        self.weight_logits = nn.Parameter(torch.zeros(target_dim, num_embedding_types))
        self.skip_init = True  # Skip initialization in param_init_fn

    def forward(self, embeddings_list: List[Tensor]) -> Tensor:
        """Apply softmax-normalized weights to combine embeddings."""
        if len(embeddings_list) == 1:
            return embeddings_list[0]

        # Stack embeddings along a new dimension
        stacked_embeddings = torch.stack(embeddings_list, dim=2)

        # Apply softmax to weights across embedding types for each dimension
        # This ensures weights for each dimension sum to 1
        softmax_weights = torch.nn.functional.softmax(self.weight_logits, dim=1)

        # Apply weighted combination
        result = torch.einsum("bsed,de->bsd", stacked_embeddings, softmax_weights)

        return result


class ConcatAndProjectCombination(nn.Module):
    """Combines embeddings by concatenation followed by projection."""

    def __init__(
        self,
        target_dim: int,
        num_embedding_types: int,
        mlp_config: Optional[Dict] = None,
    ):
        super().__init__()
        # Calculate concatenated dimension (all embeddings are target_dim)
        concat_dim = num_embedding_types * target_dim

        # Check if MLP configuration is provided
        if mlp_config is not None:
            # Validate MLP config
            required_fields = ["hidden_layers", "activation", "use_layer_norm"]
            missing_fields = [
                field for field in required_fields if field not in mlp_config
            ]

            if missing_fields:
                log.warning(
                    f"MLP config missing required fields: {missing_fields}. "
                    f"Falling back to simple linear projection.",
                )
                # Fall back to simple linear projection
                self.projection = nn.Linear(concat_dim, target_dim, bias=True)
            else:
                # Build MLP
                self.projection = self._build_mlp(concat_dim, target_dim, mlp_config)
        else:
            # No MLP config, use simple linear projection (backward compatible)
            self.projection = nn.Linear(concat_dim, target_dim, bias=True)

        log.info("Architectue of ConcatAndProjectCombination:")
        log.info(self.projection)

    def _build_mlp(
        self,
        input_dim: int,
        output_dim: int,
        mlp_config: Dict,
    ) -> nn.Sequential:
        """Build an MLP according to `mlp_config`.

        * If `resolve_ffn_act_fn` returns an `nn.Module`, we append it directly.
        * If it returns a functional op (e.g. `F.relu`), we wrap it in a tiny
          `nn.Module` so it plays nicely inside `nn.Sequential`.
        """
        layers: list[nn.Module] = []

        # Parse hidden layer spec (int or "512,256,128")
        hidden_spec = mlp_config["hidden_layers"]
        if isinstance(hidden_spec, int):
            hidden_dims = [hidden_spec]
        elif isinstance(hidden_spec, str):
            hidden_dims = [int(x) for x in hidden_spec.split(",") if x.strip()]
        else:
            log.warning(
                f"Invalid hidden_layers spec {hidden_spec!r}; "
                "falling back to a single Linear projection.",
            )
            return nn.Sequential(nn.Linear(input_dim, output_dim, bias=False))

        activation_name = mlp_config["activation"]
        use_layer_norm = mlp_config["use_layer_norm"]

        current_dim = input_dim
        for hidden_dim in hidden_dims:
            # Linear
            layers.append(nn.Linear(current_dim, hidden_dim, bias=False))

            # Activation
            act_obj = resolve_ffn_act_fn({"name": activation_name})

            if isinstance(act_obj, nn.Module):
                layers.append(act_obj)
            elif callable(act_obj):

                class _FunctionalAct(nn.Module):
                    def __init__(self, fn):
                        super().__init__()
                        self.fn = fn

                    def forward(self, x):
                        return self.fn(x)

                layers.append(_FunctionalAct(act_obj))
            else:
                raise TypeError(
                    f"Unsupported activation object returned for "
                    f"'{activation_name}': {type(act_obj)}",
                )

            # Optional layer norm
            if use_layer_norm:
                layers.append(nn.LayerNorm(hidden_dim))

            current_dim = hidden_dim

        # Final projection to `output_dim`
        layers.append(nn.Linear(current_dim, output_dim, bias=False))
        return nn.Sequential(*layers)

    def forward(self, embeddings_list: List[Tensor]) -> Tensor:
        """Concatenate embeddings and project back to target dimension."""
        if len(embeddings_list) == 1:
            return embeddings_list[0]

        # Concatenate along the feature dimension
        concatenated = torch.cat(embeddings_list, dim=-1)

        # Project back to target_dim
        return self.projection(concatenated)


class GateAndSumCombination(nn.Module):
    """Combines embeddings using learned gating values."""

    def __init__(self, target_dim: int, num_embedding_types: int):
        super().__init__()
        # Create a gate generator - produces values between 0 and 1
        self.gate_generator = nn.Sequential(
            nn.Linear(target_dim, num_embedding_types),
            nn.Sigmoid(),
        )

    def forward(self, embeddings_list: List[Tensor]) -> Tensor:
        """Apply dynamic gates based on content to combine embeddings."""
        if len(embeddings_list) == 1:
            return embeddings_list[0]

        # Stack embeddings: [B, S, num_emb, D]
        stacked = torch.stack(embeddings_list, dim=2)
        batch_size, seq_len, num_emb, dim = stacked.shape

        # Use the first embedding to generate gates
        gates = self.gate_generator(embeddings_list[0])  # [B, S, num_emb]
        gates = gates.unsqueeze(-1)  # [B, S, num_emb, 1]

        # Apply gates and sum
        gated_embeddings = stacked * gates
        result = gated_embeddings.sum(dim=2)  # [B, S, D]

        return result


class GeneEncoder(nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        target_dim: int,
        padding_idx: Optional[int] = None,
        use_norm: bool = False,
        gene_embedding_config: Optional[Dict] = None,
    ):
        super().__init__()
        self.use_norm = use_norm
        self.target_dim = target_dim

        # Track embeddings and their types
        self.embedding_types = []

        # Initialize the primary trainable embedding if specified
        use_trainable = True
        if gene_embedding_config is not None:
            use_trainable = gene_embedding_config.get("trainable", True)

        if use_trainable:
            self.primary_embedding = nn.Embedding(
                num_embeddings,
                target_dim,
                padding_idx=padding_idx,
            )
            self.embedding_types.append("primary")
            log.info(
                f"Initialized trainable primary embedding with dimension {target_dim}",
            )
        else:
            self.primary_embedding = None

        # Initialize additional embeddings if specified
        self.additional_embeddings = nn.ModuleDict()

        if gene_embedding_config is not None and "embeddings" in gene_embedding_config:
            embeddings_config = gene_embedding_config.get("embeddings", {})
            for emb_name, emb_config in embeddings_config.items():
                try:
                    # Load the embedding file
                    emb_path = emb_config.get("local")
                    if emb_path is None:
                        log.warning(f"No local path specified for {emb_name}, skipping")
                        continue

                    state_dict = torch.load(emb_path, weights_only=True)

                    if "embedding.weight" in state_dict:
                        pretrained_weight = state_dict["embedding.weight"]
                        emb_dim = pretrained_weight.shape[1]
                        pretrained_vocab_size = pretrained_weight.shape[0]
                        log.info(
                            f"Loaded {emb_name} embeddings with dimension {emb_dim}",
                        )

                        # Create embedding layer
                        self.additional_embeddings[f"{emb_name}_embedding"] = (
                            nn.Embedding(
                                num_embeddings,
                                emb_dim,
                                padding_idx=padding_idx,
                            )
                        )

                        # Load the weights and handle dimension mismatch
                        if pretrained_vocab_size != num_embeddings:
                            log.info(
                                f"{emb_name} embedding size mismatch: pretrained has {pretrained_vocab_size} tokens, "
                                f"but model expects {num_embeddings} tokens. Loading pretrained weights to top rows.",
                            )
                            # Create new weight tensor with correct size
                            new_weight = torch.zeros(
                                num_embeddings,
                                emb_dim,
                                dtype=pretrained_weight.dtype,
                            )
                            # Copy pretrained weights to top portion
                            min_vocab = min(pretrained_vocab_size, num_embeddings)
                            new_weight[:min_vocab] = pretrained_weight[:min_vocab]
                            # Load the adjusted weights
                            self.additional_embeddings[
                                f"{emb_name}_embedding"
                            ].load_state_dict(
                                {"weight": new_weight},
                            )
                        else:
                            # Load the weights directly
                            self.additional_embeddings[
                                f"{emb_name}_embedding"
                            ].load_state_dict(
                                {"weight": pretrained_weight},
                            )

                        # Debug: Verify that extra rows are zero
                        if pretrained_vocab_size != num_embeddings:
                            loaded_weight = self.additional_embeddings[
                                f"{emb_name}_embedding"
                            ].weight.data
                            # Check if bottom rows are zero
                            if pretrained_vocab_size < num_embeddings:
                                extra_rows = loaded_weight[pretrained_vocab_size:]
                                log.info(
                                    f"{emb_name}: Extra {num_embeddings - pretrained_vocab_size} rows - "
                                    f"all zeros: {torch.all(extra_rows == 0).item()}, "
                                    f"max value: {extra_rows.abs().max().item()}",
                                )
                            # Log a sample of indices around the boundary
                            log.info(
                                f"{emb_name}: Sample embeddings around vocab boundary:",
                            )
                            for idx in range(
                                max(0, pretrained_vocab_size - 2),
                                min(num_embeddings, pretrained_vocab_size + 3),
                            ):
                                if idx < num_embeddings:
                                    log.info(
                                        f"  Token {idx}: norm={loaded_weight[idx].norm().item():.4f}",
                                    )

                        # Set trainable/fixed
                        fix_embedding = emb_config.get("fix_embedding", True)
                        self.additional_embeddings[
                            f"{emb_name}_embedding"
                        ].weight.requires_grad = not fix_embedding
                        log.info(
                            f"{emb_name} embeddings are {'fixed' if fix_embedding else 'trainable'}",
                        )

                        # Always add a projection (linear rotation even if they match)
                        self.additional_embeddings[f"{emb_name}_proj"] = nn.Linear(
                            emb_dim,
                            target_dim,
                            bias=False,
                        )
                        log.info(
                            f"Added projection for {emb_name} from {emb_dim} to {target_dim}",
                        )

                        # Add normalization if requested
                        use_emb_norm = emb_config.get("use_norm", False)
                        if use_emb_norm:
                            self.additional_embeddings[f"{emb_name}_norm"] = (
                                nn.LayerNorm(target_dim)
                            )
                            log.info(f"Added layer normalization for {emb_name}")

                        # Mark to skip initialization
                        self.additional_embeddings[
                            f"{emb_name}_embedding"
                        ].skip_init = True

                        # Track this embedding type
                        self.embedding_types.append(emb_name)
                    else:
                        log.warning(
                            f"No embedding.weight found in {emb_path} for {emb_name}",
                        )
                except Exception as e:
                    log.error(f"Failed to load {emb_name} embeddings: {e!s}")

        # Check if we have any embeddings at all
        if not self.primary_embedding and not self.embedding_types:
            log.warning("No embeddings configured, falling back to trainable embedding")
            self.primary_embedding = nn.Embedding(
                num_embeddings,
                target_dim,
                padding_idx=padding_idx,
            )
            self.embedding_types.append("primary")

        # Create the combination strategy based on configuration
        num_embeddings_total = len(self.embedding_types)

        # Default combination strategy
        combination_strategy = "learned_weighted"

        # Get combination strategy from config if provided
        if (
            gene_embedding_config is not None
            and "combination_strategy" in gene_embedding_config
        ):
            combination_strategy = gene_embedding_config["combination_strategy"]

        if num_embeddings_total > 1:
            log.info(
                f"Using '{combination_strategy}' combination strategy for {num_embeddings_total} embeddings",
            )

            # Create the appropriate combiner based on the strategy
            if combination_strategy == "elementwise_sum":
                self.combiner = ElementwiseSumCombination(
                    target_dim,
                    num_embeddings_total,
                )
            elif combination_strategy == "learned_weighted":
                self.combiner = LearnedWeightedCombination(
                    target_dim,
                    num_embeddings_total,
                )
            elif combination_strategy == "softmax_weighted":
                self.combiner = SoftmaxWeightedCombination(
                    target_dim,
                    num_embeddings_total,
                )
            elif combination_strategy == "concat_and_project":
                # Extract MLP config if it exists
                mlp_config = gene_embedding_config.get("mlp", None)
                self.combiner = ConcatAndProjectCombination(
                    target_dim,
                    num_embeddings_total,
                    mlp_config=mlp_config,
                )
            elif combination_strategy == "gate_and_sum":
                self.combiner = GateAndSumCombination(target_dim, num_embeddings_total)
            else:
                log.warning(
                    f"Unknown combination strategy '{combination_strategy}', falling back to learned_weighted",
                )
                self.combiner = LearnedWeightedCombination(
                    target_dim,
                    num_embeddings_total,
                )
        else:
            self.combiner = None
            log.info("Only one embedding type, no combination needed")

        # Global normalization
        if self.use_norm:
            self.enc_norm = nn.LayerNorm(target_dim)

    def forward(self, x: Tensor) -> Tensor:
        # Collect all embeddings
        all_embeddings = []

        # Process each embedding type
        for i, emb_type in enumerate(self.embedding_types):
            if emb_type == "primary":
                if self.primary_embedding is not None:
                    emb_tensor = self.primary_embedding(x)
                    all_embeddings.append(emb_tensor)
            else:
                emb_tensor = self.additional_embeddings[f"{emb_type}_embedding"](x)

                # Apply projection if it exists
                if f"{emb_type}_proj" in self.additional_embeddings:
                    emb_tensor = self.additional_embeddings[f"{emb_type}_proj"](
                        emb_tensor,
                    )

                # Apply normalization if it exists
                if f"{emb_type}_norm" in self.additional_embeddings:
                    emb_tensor = self.additional_embeddings[f"{emb_type}_norm"](
                        emb_tensor,
                    )

                all_embeddings.append(emb_tensor)

        # If only one embedding, return it directly
        if len(all_embeddings) == 1:
            result = all_embeddings[0]
        else:
            # Combine embeddings using the selected strategy
            result = self.combiner(all_embeddings)

        # Apply global normalization if specified
        if self.use_norm:
            result = self.enc_norm(result)

        return result


class ChemEncoder(nn.Module):
    def __init__(
        self,
        drug_fps_path: dict,
        d_out: int,
        padding_idx: int = 0,
        activation: str = "leaky_relu",
        use_norm: bool = True,
        freeze: bool = False,
    ):
        super().__init__()

        # download pretrained drug embeddings - morgan fingerprints
        if dist.get_local_rank() == 0:
            download_file_from_s3_url(
                s3_url=drug_fps_path["remote"],
                local_file_path=drug_fps_path["local"],
            )
        with dist.local_rank_zero_download_and_wait(drug_fps_path["local"]):
            dist.barrier()

        drug_fps = torch.as_tensor(np.load(drug_fps_path["local"]), dtype=torch.float32)
        embedding_dim = drug_fps.shape[1]

        self.embedding = nn.Embedding.from_pretrained(
            drug_fps,
            padding_idx=padding_idx,
            freeze=freeze,
        )
        self.fc = nn.Linear(embedding_dim, d_out)
        self.activation = resolve_ffn_act_fn({"name": activation})
        self.proj = nn.Linear(d_out, d_out)

        self.use_norm = use_norm
        if self.use_norm:
            self.norm = nn.LayerNorm(d_out)

    def forward(self, x: Tensor) -> Tensor:
        x = self.embedding(x)  # (batch, d_out)
        x = self.activation(self.fc(x))
        x = self.proj(x)  # (batch, d_out)

        if self.use_norm:
            x = self.norm(x)
        return x


class ContinuousValueEncoder(nn.Module):
    """Encode real number values to a vector using neural nets projection."""

    def __init__(
        self,
        d_model: int,
        dropout: float = 0.1,
        max_value: int = 512,
        activation: str = "relu",
        use_norm: bool = False,
    ):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.linear1 = nn.Linear(1, d_model)
        self.activation = resolve_ffn_act_fn({"name": activation})
        self.linear2 = nn.Linear(d_model, d_model)
        self.use_norm = use_norm
        if self.use_norm:
            self.norm = nn.LayerNorm(d_model)
        self.max_value = max_value

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, seq_len]
        """
        # expand last dimension
        x = x.unsqueeze(-1)
        # clip x to [-inf, max_value]
        x = torch.clamp(x, max=self.max_value)
        x = self.activation(self.linear1(x))
        x = self.linear2(x)
        if self.use_norm:
            x = self.norm(x)
        return self.dropout(x)


class CategoryValueEncoder(nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: Optional[int] = None,
        use_norm: bool = False,
    ):
        super().__init__()
        self.embedding = nn.Embedding(
            num_embeddings,
            embedding_dim,
            padding_idx=padding_idx,
        )
        self.use_norm = use_norm
        if self.use_norm:
            self.enc_norm = nn.LayerNorm(embedding_dim)

    def forward(self, x: Tensor) -> Tensor:
        x = x.long()
        x = self.embedding(x)  # (batch, seq_len, embsize)
        if self.use_norm:
            x = self.enc_norm(x)
        return x


class ExprDecoder(nn.Module):
    """Consists of three linear functions and leaky-relu as an activation
    function."""

    def __init__(
        self,
        d_model: int,
        n_outputs: int = 1,
        n_layers: int = 2,
        activation: str = "leaky_relu",
    ):
        super().__init__()
        d_in = d_model
        self.activation = resolve_ffn_act_fn({"name": activation})
        self.linear_layers = nn.ModuleList(
            [nn.Linear(d_in, d_model) for _ in range(n_layers)],
        )
        self.out_proj = nn.Linear(d_model, n_outputs)

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        """X is the output of the transformer, (batch, seq_len, d_model)"""
        for layer in self.linear_layers:
            x = self.activation(layer(x))
        pred_value = self.out_proj(x)  # (batch, seq_len, n_outputs)
        if pred_value.shape[-1] == 1:
            pred_value = pred_value.squeeze(-1)
        return {"pred": pred_value}


class AffineExprDecoder(torch.nn.Module):
    def __init__(
        self,
        d_model: int,
        explicit_zero_prob: bool = False,
        activation: Optional[str] = None,
        tanh_coeff: bool = False,
        adaptive_bias: bool = False,
    ):
        """Predict the expression value of each gene in an affine like form of
        Ax + b. This decoder takes two ExprDecoder intrinsically to genrate the
        coefficient A and bias b.

        Args:
            d_model: The embedding dimension.
            explicit_zero_prob: If True, predict the probability of each gene being
                zero.
            activation: The activation function for the coefficient A and bias b.
            tanh_coeff: If True, use tanh activation for the coefficient A.
            adaptive_bias: If True, use a learnable bias for the bias b.
        """
        super().__init__()
        self.explicit_zero_prob = explicit_zero_prob
        self.tanh_coeff = tanh_coeff
        self.adaptive_bias = adaptive_bias
        self.coeff_decoder = ExprDecoder(d_model)
        self.bias_decoder = ExprDecoder(d_model)

        self.activation = activation
        if activation is not None:
            assert hasattr(nn, activation), f"Unknown activation: {activation}"
            self.activation = getattr(nn, activation)()

    def forward(self, x: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embsize]
            values: Tensor, shape [batch_size, seq_len]

        Returns:
            output Tensor of shape [batch_size, seq_len]
        """
        coeff = self.coeff_decoder(x)
        bias = self.bias_decoder(x)

        if self.activation is not None:
            coeff["pred"] = self.activation(coeff["pred"])
            bias["pred"] = self.activation(bias["pred"])

        if self.adaptive_bias:
            non_zero_value_mean = values.sum(dim=1, keepdim=True) / (values != 0).sum(
                dim=1,
                keepdim=True,
            )
            bias["pred"] = bias["pred"] * non_zero_value_mean

        if self.explicit_zero_prob:
            return {
                "pred": coeff["pred"] * values + bias["pred"],
                "zero_probs": coeff["zero_probs"],
            }

        return {"pred": coeff["pred"] * values + bias["pred"]}


class MVCDecoder(nn.Module):
    """Decoder for the masked value prediction for cell embeddings."""

    def __init__(
        self,
        d_model: int,
        arch_style: str = "inner product",
        query_activation: str = "sigmoid",
        scaled_dot_product: bool = False,
    ) -> None:
        """
        Args:
            d_model (:obj:`int`): dimension of the gene embedding.
            arch_style (:obj:`str`): architecture style of the decoder, choice from
                1. "inner product" or 2. "concat query" or 3. "sum query".
            query_activation (:obj:`nn.Module`): activation function for the query
                vectors.
            hidden_activation (:obj:`nn.Module`): activation function for the hidden
                layers.
        """
        super().__init__()
        d_in = d_model
        self.scaled_dot_product = scaled_dot_product
        if arch_style == "inner product":
            self.gene2query = nn.Linear(d_model, d_model)
            self.query_activation = resolve_ffn_act_fn({"name": query_activation})
            self.W = nn.Linear(d_model, d_in, bias=False)
        else:
            raise ValueError(f"Unknown arch_style: {arch_style}")

        self.arch_style = arch_style

    def forward(
        self,
        cell_emb: Tensor,
        gene_embs: Tensor,
    ) -> Union[Tensor, Dict[str, Tensor]]:
        """
        Args:
            cell_emb: Tensor, shape (batch, embsize=d_model)
            gene_embs: Tensor, shape (batch, seq_len, embsize=d_model)
        """
        if self.arch_style == "inner product":
            query_vecs = self.query_activation(
                self.gene2query(gene_embs),
            )  # (batch, seq_len, embsize)
            inner_product_dimension = query_vecs.shape[-1]
            cell_emb = cell_emb.unsqueeze(2)  # (batch, embsize, 1)
            pred_value = torch.bmm(self.W(query_vecs), cell_emb).squeeze(
                2,
            )  # (batch, seq_len)
            if self.scaled_dot_product:
                pred_value = pred_value / torch.sqrt(
                    torch.tensor(inner_product_dimension, dtype=pred_value.dtype),
                )
            return {"pred": pred_value}
        else:
            raise ValueError(f"Unknown arch_style: {self.arch_style}")
