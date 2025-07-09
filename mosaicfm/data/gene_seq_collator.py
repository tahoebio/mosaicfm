# Copyright (C) Vevo Therapeutics 2024-2025. All rights reserved.
import json
import logging
log = logging.getLogger(__name__)

from typing import Dict, List, Mapping, Optional, Tuple, Union

import numpy as np
import torch
from composer.utils import dist
from transformers import DefaultDataCollator

from mosaicfm.tokenizer import GeneVocab
from mosaicfm.utils import download_file_from_s3_url
from mosaicfm.data.collator import log_transform, binning



class GeneSeqCollator(DefaultDataCollator):
    """Data collator for the mask value learning task. It pads the sequences to
    the maximum length in the batch and masks the gene expression values.

    Args:
        vocab (:obj: GeneVocab): The vocabulary that includes the gene ids, name, special tokens, etc.
        use_chem_token (:obj:`bool`): whether to create and use the chemical token in the sequence.
        drug_to_id_path (:obj:`dict`): path to the drug to id .json file.
        pad_token_id (:obj:`int`, optional): the token id to use for padding.
            This is required if do_padding is True.
        pad_value (:obj:`int`): the value to use for padding the expression
            values to the max length.
        target_sum (:obj:`int`): The target sum of the normalized counts before log1p transformation.
        max_length (:obj:`int`, optional): the maximum length of the sequences.
            This is required if do_padding is True.
        num_high_exp_genes (:obj:`int`): the number of highly expressed genes to keep.
        reserve_keys (:obj:`List[str]`, optional): a list of keys in the examples
            to reserve in the output dictionary. Default to []. These fields
            will be kept unchanged in the output.
        keep_first_n_tokens (:obj:`int`): the number of tokens in the beginning
            of the sequence to keep unchanged from sampling. This is useful when
            special tokens have been added to the beginning of the sequence.
            Default to 1.
        is_train (:obj:`bool`): whether the data is for training or not.
        num_bins (:obj:`int`): the number of bins to use for binning the expression
        right_binning (:obj:`bool`): whether to use right sided-binning. Torch default is False
    """

    def __init__(
        self,
        vocab: GeneVocab,
        drug_to_id_path: Optional[dict] = None,
        use_chem_token: int = False,
        pad_token_id: Optional[int] = None,
        pad_value: int = 0,
        target_sum: int = 10000,
        max_length: Optional[int] = None,
        num_high_exp_genes: int = 512,
        reserve_keys: Optional[List[str]] = None,
        keep_first_n_tokens: int = 1,
        is_train: bool = True,
        num_bins: int = 51,
        right_binning: bool = False,
        return_tensors: str = "pt",
    ):
        super().__init__(return_tensors=return_tensors)
        self.vocab = vocab
        self.use_chem_token = use_chem_token
        self.pad_token_id = pad_token_id
        self.pad_value = pad_value
        self.target_sum = target_sum
        self.max_length = max_length
        self.num_high_exp_genes = num_high_exp_genes
        self.reserve_keys = reserve_keys if reserve_keys is not None else []
        self.keep_first_n_tokens = keep_first_n_tokens
        self.is_train = is_train
        self.num_bins = num_bins
        self.right_binning = right_binning

        # filter non_special gene_ids
        gene_to_id = vocab.get_stoi()
        self.non_special_gene_ids = torch.tensor(
            [
                gene_id
                for gene_name, gene_id in gene_to_id.items()
                if not gene_name.startswith("<")
            ],
        )
        if self.use_chem_token:
            assert "<drug>" in vocab, "<drug> token must be in the vocabulary."
            self.drug_token_id = vocab["<drug>"]
        else:
            self.drug_token_id = None
        assert not self.use_chem_token or drug_to_id_path is not None, (
            "If `use_chem_token` is True, `drug_to_id_path` must be provided.",
        )
        assert drug_to_id_path is None or self.use_chem_token, (
            "If `drug_to_id_path` is provided, `use_chem_token` must be True.",
        )
        assert not self.use_chem_token or self.keep_first_n_tokens > 1, (
            "If `use_chem_token` is True, we need to keep <cls> and <drug> token in the beggining of pcpt_genes. So `keep_first_n_tokens` must be >=2!",
        )
        # load drug_to_id mapping if present
        if self.use_chem_token:
            if dist.get_local_rank() == 0:
                download_file_from_s3_url(
                    s3_url=drug_to_id_path["remote"],
                    local_file_path=drug_to_id_path["local"],
                )
            with dist.local_rank_zero_download_and_wait(drug_to_id_path["local"]):
                dist.barrier()

            with open(drug_to_id_path["local"]) as f:
                self.drug_to_id = json.load(f)

    def __post_init__(self):
        if self.pad_token_id is None:
            raise ValueError("`pad_token_id` is required.")
        if self.max_length is None:
            raise ValueError("`max_length` is required.")

        if isinstance(self.reserve_keys, str):
            self.reserve_keys = [self.reserve_keys]

        if self.keep_first_n_tokens < 0 or self.keep_first_n_tokens > self.max_length:
            raise ValueError(
                "`keep_first_n_tokens` must be between 0 and `max_length` "
                f"({self.max_length}).",
            )

    def __call__(
        self,
        examples: List[Dict[str, torch.Tensor]],
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            examples (:obj:`List[Dict[str, torch.Tensor]]`): a list of data dicts.
                Each dict is for one cell. It contains multiple 1 dimensional tensors
                like the following exmaple:
                    {'id': tensor(184117),
                    'genes': tensor([36572, 17868, ..., 17072]),
                    'expressions': tensor([ 0.,  2., ..., 18.])}

        Returns:
            :obj:`Dict[str, torch.Tensor]`: a dict of tensors.
        """
        for example in examples:
            if self.use_chem_token:
                drug = (
                    example["drug"]
                    if "drug" in example and example["drug"] in self.drug_to_id
                    else "<pad>"
                )
                example["drug_id"] = torch.as_tensor(
                    self.drug_to_id[drug],
                    dtype=torch.int,
                )
            if isinstance(example["genes"], list):
                example["genes"] = torch.as_tensor(example["genes"])
            example["genes"] = torch.squeeze(example["genes"])
            if isinstance(example["expressions"], list):
                example["expressions"] = torch.as_tensor(example["expressions"])
            example["expressions"] = torch.squeeze(example["expressions"])
        if len(self.reserve_keys) > 0:
            assert all(key in examples[0] for key in self.reserve_keys), (
                f"reserve_keys must be a subset of the keys in the examples. "
                f"Got {self.reserve_keys} but expected keys in {list(examples[0].keys())}."
            )

        if self.is_train:
            data_dict = self._call_train(examples)
        else:
            raise NotImplementedError("SEDataCollator is not implemented for evaluation.")

        # add reserved keys
        device = examples[0]["genes"].device
        for key in self.reserve_keys:
            data_ = [example[key] for example in examples]
            if isinstance(data_[0], torch.Tensor):
                # if the reserved key is a tensor, stack them
                data_dict[key] = torch.stack(data_, dim=0).to(device)
            else:
                data_dict[key] = data_  # if not tensor, just keep the list

        return data_dict

    
    def _call_train(
        self,       
        examples: List[Dict[str, torch.Tensor]],
    ) -> Dict[str, torch.Tensor]:
        """

        Each example is like:
            {'id': tensor(184117),
            'genes': tensor([36572, 17868, ..., 17072]),
            'expressions': tensor([ 0.,  2., ..., 18.])},
            'drug_id': Optinal = tensor(256), id 0 refers to <pad> token and indicates that drug is not available


        Returns:
            Dict[str, torch.Tensor]: a dict of tensors.
            Example:
                {'in_gene_ids': tensor([[36572, 17868, ..., 17072],
                                        [36572, 17868, ..., 17072],
                                        ...,
                                        [36572, 17868, ..., 17072]]),
                                        
                'in_exprs': tensor([[ 0.,  2., ..., 18.], #binned
        
                'high_exp_gene_ids': tensor([[36573, 17869, ..., 17073],
                'high_exp_exprs': tensor([[ 1.,  3., ..., 19.],  #log_transformed

                'non_exp_gene_ids': tensor([[36574, 17870, ..., 17074],
                'non_exp_exprs': tensor([[ 0, 0., ...,  0.],  #log_transformed

                'rand_gene_ids': tensor([[36575, 17871, ..., 17075],
                'rand_exprs': tensor([[ 0.1, 0.15, ...,  0.75]]), #log_transformed
                }
        """


        in_gene_ids = []
        in_exprs = []
        high_exp_gene_ids = []
        high_exp_exprs = []
        non_exp_gene_ids = []
        non_exp_exprs = []
        rand_exprs = []
        drug_ids = [] if self.use_chem_token else None

        device = examples[0]["genes"].device

        #1) We need to choose the random genes for the whole batch
        num_rand_genes = self.num_high_exp_genes // 2
        idx = torch.randperm(len(self.non_special_gene_ids), device=device)[:num_rand_genes]
        rand_gene_id = self.non_special_gene_ids[idx]

        for i in range(len(examples)):
            genes = examples[i]["genes"]
            expressions = examples[i]["expressions"]
            assert len(genes) == len(expressions), (
                f"Genes and expressions must have the same length. "
                f"Got {genes.shape} genes and {expressions.shape} expressions.",
            )

            # we need to filter out the samples with less than num_high_exp_genes genes
            if self.num_high_exp_genes > len(genes):
                log.warning(
                    f"Number of highly expressed genes ({self.num_high_exp_genes}) is greater than the number of genes ({len(genes)}). "
                    "Skipping this example.",
                )
                continue

            if self.use_chem_token:
                #2) add drug token <drug>, and pad_value=-2 expression at location 1  (after <cls>) of genes and expressions
                genes = torch.cat(
                    (
                        genes[:1],
                        torch.tensor(
                            [self.drug_token_id],
                            device=genes.device,
                            dtype=genes.dtype,
                        ),
                        genes[1:],
                    ),
                )
                expressions = torch.cat(
                    (
                        expressions[:1],
                        torch.tensor(
                            [self.pad_value],
                            device=expressions.device,
                            dtype=expressions.dtype,
                        ),
                        expressions[1:],
                    ),
                )


            #3) sorth genes based on expression values
            order = torch.argsort(expressions[self.keep_first_n_tokens:], descending=True)
            expressions = torch.cat((expressions[:self.keep_first_n_tokens], expressions[self.keep_first_n_tokens:][order]), dim=0)
            log_expressions = expressions.clone()  # make a copy for log transformation
            genes = torch.cat((genes[:self.keep_first_n_tokens], genes[self.keep_first_n_tokens:][order]), dim=0)

            assert len(genes) == len(expressions), (
                f"Genes and expressions must have the same length after sorting. "
                f"Got {genes.shape} genes and {expressions.shape} expressions.",
            )

            #4) log transform and bin the expressions for input and output
            slice_exp = expressions[self.keep_first_n_tokens :].clone()
            expressions[self.keep_first_n_tokens:] = binning(
                row=slice_exp,
                n_bins=self.num_bins,
                right=self.right_binning,
            ) #binned expression include all of genes including the first n tokens (special tokens)

            slice_log_Exp = log_expressions[self.keep_first_n_tokens :].clone()
            log_expressions[self.keep_first_n_tokens:] = log_transform(
                row=slice_log_Exp,
                target_sum=self.target_sum,
            ) # log expression include all of genes including the first n tokens (special tokens)

            assert len(genes) == len(expressions) == len(log_expressions), (
                f"Genes, binned expressions and log expressions must have the same length. "
                f"Got {genes.shape} genes, {expressions.shape} expressions and {log_expressions.shape} log expressions.",
            )

            #5) create input to the model which consists of 2048 highly expressed genes and their expressions
            if len(genes)>= self.max_length: 
                # truncate the genes and expressions to max_length
                in_gene_id = genes[: self.max_length]
                in_expr = expressions[: self.max_length]
            else: 
                # add unexpressed genes to the genes and expressions to max_length
                in_gene_id, in_expr = self._pad_unexp_genes(
                        genes,
                        expressions,
                        max_length=self.max_length,
                    )
            
            assert len(in_gene_id) == self.max_length, (
                f"Input genes must have the same length as max_length. "
                f"Got {in_gene_id.shape} genes and max_length {self.max_length}.",
            )


            #6) create labels (output) of the model which in default consists of 512 highly expressed genes, 512 non expressed genes and 256 random genes
            
            # 6.1) get the highly expressed genes and their expressions
            high_exp_expr = log_expressions[: self.num_high_exp_genes]
            high_exp_gene_id = genes[: self.num_high_exp_genes]

            assert len(high_exp_gene_id) == self.num_high_exp_genes, (
                f"High expressed genes must have the same length as num_high_exp_genes. "
                f"Got {high_exp_gene_id.shape} genes and num_high_exp_genes {self.num_high_exp_genes}.",
            )

            # 6.2) get the non-expressed genes and their expressions
            num_non_exp_genes = self.num_high_exp_genes
            non_exp_gene_id, non_exp_expr =self._pick_unexp_genes(
                genes[self.keep_first_n_tokens:],
                k = num_non_exp_genes,
                type = expressions.dtype,

            )

            assert len(non_exp_gene_id) == self.num_high_exp_genes, (
                f"Non expressed genes must have the same length as num_high_exp_genes. "
                f"Got {non_exp_gene_id.shape} genes and num_high_exp_genes {num_non_exp_genes}.",
            )

            assert torch.isin(non_exp_gene_id, genes).sum() == 0, (    
                f"Non expressed genes must not overlap with expressed genes. "
            )

            # 6.3) get expressions for the random genes
            rand_expr = self._collect_expr_rand_genes(
                genes=genes[self.keep_first_n_tokens:],
                expressions=log_expressions[self.keep_first_n_tokens:],
                rand_genes=rand_gene_id
            )

            assert len(rand_expr) == len(rand_gene_id), (
                f"Random genes and their expressions must have the same length, and be half the number of high expressed genes."
                f"Got {rand_expr.shape} expressions and {rand_gene_id.shape} random genes. Number of random genes should be {self.num_high_exp_genes // 2}.",
            )
 
            #7) append the data to the lists

            in_gene_ids.append(in_gene_id)
            in_exprs.append(in_expr)
            high_exp_gene_ids.append(high_exp_gene_id)
            high_exp_exprs.append(high_exp_expr)
            non_exp_gene_ids.append(non_exp_gene_id)
            non_exp_exprs.append(non_exp_expr)
            rand_exprs.append(rand_expr)

            if self.use_chem_token:
                # add drug id, id=0 corresponds to <pad> which indicates that drug is not available
                drug = examples[i]["drug_id"]
                drug_ids.append(drug)

        
        #8) stack the lists to tensors
        in_gene_ids = torch.stack(in_gene_ids, dim=0)
        in_exprs = torch.stack(in_exprs, dim=0)
        high_exp_gene_ids = torch.stack(high_exp_gene_ids, dim=0)
        high_exp_exprs = torch.stack(high_exp_exprs, dim=0)
        non_exp_gene_ids = torch.stack(non_exp_gene_ids, dim=0)
        non_exp_exprs = torch.stack(non_exp_exprs, dim=0)
        rand_exprs = torch.stack(rand_exprs, dim=0)
        rand_gene_ids = rand_gene_id.unsqueeze(0).repeat(rand_exprs.shape[0], 1)


        if self.use_chem_token:
            drug_ids = torch.stack(drug_ids)

            data_dict = {
                "in_gene_ids": in_gene_ids,
                "in_exprs": in_exprs,
                "high_exp_gene_ids": high_exp_gene_ids,
                "high_exp_exprs": high_exp_exprs,
                "non_exp_gene_ids": non_exp_gene_ids,
                "non_exp_exprs": non_exp_exprs,
                "rand_exprs": rand_exprs,
                "rand_gene_ids": rand_gene_ids,
                "drug_ids": drug_ids,
            }
        else:
            data_dict = {
                "in_gene_ids": in_gene_ids,
                "in_exprs": in_exprs,
                "high_exp_gene_ids": high_exp_gene_ids,
                "high_exp_exprs": high_exp_exprs,
                "non_exp_gene_ids": non_exp_gene_ids,
                "non_exp_exprs": non_exp_exprs,
                "rand_exprs": rand_exprs,
                "rand_gene_ids": rand_gene_ids,
            }
        return data_dict

    def _pad_unexp_genes(
        self,
        *arrays: torch.Tensor,  # First tensor is genes, rest are  expressions respectively processed expressions, raw expressions (optional)
        max_length: int,
    ):

        """Pad sequence with unexpressed genes. """

        device = arrays[0].device

        num_to_pad = max_length - len(arrays[0])


        random_unexp_genes, random_unexp_expressions = self._pick_unexp_genes(
            arrays[0], k=num_to_pad, type=arrays[1].dtype 
        )  # pick unexpressed genes and their expressions

        # Pad the first tensor(gene_ids) with random unexpressed gene ids and the rest (expressions) with zeros.
        return tuple(
            torch.cat(
                [
                    array,
                    (
                        random_unexp_genes
                        if i == 0
                        else random_unexp_expressions
                    ),
                ],
            )
            for i, array in enumerate(arrays)
        )


    def _pick_unexp_genes(
        self,
        genes: torch.Tensor,
        k: int,
        type: Union[torch.dtype, str] = torch.float32,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Pick unexpressed genes. """
        device = genes.device

        # get list of all valid gene ids
        non_special_gene_ids = self.non_special_gene_ids.to(device)

        # filter out the expressed gene ids
        mask = ~torch.isin(non_special_gene_ids, genes)
        unexp_genes = non_special_gene_ids[mask]

        # randomly sample from unexpressed gene ids
        idx = torch.randperm(unexp_genes.shape[0])[:k]
        random_unexp_genes = unexp_genes[idx]
        unexp_genes_expressions = torch.zeros(
            k, dtype=type, device=device
        )  # create zero expressions for unexpressed genes


        return random_unexp_genes, unexp_genes_expressions


    def _collect_expr_rand_genes(
        self,
        genes: torch.Tensor,
        expressions: torch.Tensor,
        rand_genes: torch.Tensor,
    ) -> torch.Tensor:
        """Collect expressions for the random genes."""

        # sort gene_ids (and shuffle values the same way)
        genes,  idx_sort = torch.sort(genes)
        expressions = expressions[idx_sort]

        # for each random_gene, find the insertion‐point in sorted_ids
        pos = torch.searchsorted(genes, rand_genes)

        # clamp any “off the end” indices back into the right boundry
        pos_clamped = torch.clamp(pos, max=genes.size(0) - 1)

        # mask where the gene actually matches (handles the case of multiple right boundary in pos)
        mask = (genes[pos_clamped] == rand_genes)

        # gather expression where mask is True (common genes), else 0 
        out = torch.zeros_like(rand_genes, dtype=expressions.dtype)
        out[mask] = expressions[pos_clamped][mask]


        return out
