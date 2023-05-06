from typing import *
import json
import random

import torch
import transformers
import pytorch_lightning as pl


class TaskConfig:
    def __init__(
        self,
        task_name: Literal["paraphrase", "qa", "title-body"],
        tokenizer: transformers.AutoTokenizer,
        spec_tokens_dict: Dict,
        collator_conf: Dict,
        datasets: Dict,
        train_bs: int = 64,
        val_bs: int = 64,
        test_bs: int = 64,
    ):
        self.task_name = task_name
        self.train_bs = train_bs
        self.val_bs = val_bs
        self.test_bs = test_bs
        if task_name in ["knowledge_ground_generation"]:
            self.collator = KnowledgeExtractionCollator(
                tokenizer=tokenizer, spec_tokens_dict=spec_tokens_dict, **collator_conf
            )
        elif task_name in ["knowledge_retrieval"]:
            self.collator = KnowledgeExtractionCollator(
                tokenizer=tokenizer, spec_tokens_dict=spec_tokens_dict, **collator_conf
            )
        elif task_name in ["knowledge_extraction"]:
            self.collator = KnowledgeExtractionCollator(
                tokenizer=tokenizer, spec_tokens_dict=spec_tokens_dict, **collator_conf
            )
        elif task_name in ["answer_confirmation"]:
            self.collator = KnowledgeExtractionCollator(
                tokenizer=tokenizer, spec_tokens_dict=spec_tokens_dict, **collator_conf
            )

        self.datasets = datasets


class BaseCollator:
    def __init__(
        self,
        tokenizer,
        spec_tokens_dict: Dict,
        return_tensors: Literal["pt"] = "pt",
    ):
        self.tokenizer = tokenizer
        self.spec_tokens_dict = spec_tokens_dict
        self.return_tensors = return_tensors

        # tokenize cls
        if self.tokenizer.cls_token:
            self.cls_prefix = self.tokenizer(
                [self.tokenizer.cls_token],
                add_special_tokens=False,
                return_tensors=self.return_tensors,
            )
        else:
            self.cls_prefix = None

    def tokenize(
        self,
        prefix,
        text,
        add_special_tokens: bool = False,
        padding: Union[bool, str] = "max_length",
        truncation: Union[bool, str] = True,
        max_length: int = None,
        padding_side: Literal["left", "right"] = "right",
        truncation_side: Literal["left", "right"] = "right",
    ) -> Dict:
        self.tokenizer.padding_side = padding_side
        self.tokenizer.truncation_side = truncation_side
        if prefix is not None:
            # tokenize prefix
            prefix = self.tokenizer(
                [prefix],
                add_special_tokens=False,
                return_tensors=self.return_tensors,
            )
            # tokenize text
            text = self.tokenizer(
                text,
                add_special_tokens=add_special_tokens,
                padding=padding,
                truncation=truncation,
                max_length=max_length - 2,
                return_tensors=self.return_tensors,
            )
            # add special tokens
            if add_special_tokens and (self.cls_prefix is not None):
                text = {k: text[k][:, 1:] for k in text}

            # concatenate prefix and text
            for k in text:
                list_to_concat = [
                    prefix[k].repeat((text[k].size()[0], 1)),
                    text[k],
                ]

                if self.cls_prefix is not None:
                    list_to_concat.insert(
                        0, self.cls_prefix[k].repeat((text[k].size()[0], 1))
                    )

                text[k] = torch.concat(
                    list_to_concat,
                    dim=1,
                )
        else:
            text = self.tokenizer(
                text,
                add_special_tokens=add_special_tokens,
                padding=padding,
                truncation=truncation,
                max_length=max_length,
                return_tensors=self.return_tensors,
            )
        return text


class MultiCollator:
    def __init__(self, collators_dict):
        self.collators_dict = collators_dict

    def __call__(self, task_name, ds_name, batch) -> Dict:
        return {
            "batch": [self.collators_dict[task_name](dict(batch))],
            "type": [
                {
                    "task": task_name,
                    "source": ds_name,
                }
            ],
        }


class KnowledgeExtractionCollator(BaseCollator):
    def __init__(
        self,
        tokenizer,
        knowledge_separator: str,
        spec_tokens_dict: dict,
        inp_prefix: str,
        out_prefix: str = None,
        inp_padding_side: Literal["left", "right"] = "right",
        out_padding_side: Literal["left", "right"] = "right",
        inp_truncation_side: Literal["left", "right"] = "right",
        out_truncation_side: Literal["left", "right"] = "right",
        inp_max_len: int = 64,
        out_max_len: int = 64,
        inp_padding: Literal["max_length", True] = True,
        out_padding: Literal["max_length", True] = True,
        inp_truncation: bool = True,
        out_truncation: bool = True,
        inp_add_special_tokens: bool = True,
        out_add_special_tokens: bool = True,
        return_tensors: Literal["pt"] = "pt",
    ) -> None:
        super().__init__(
            tokenizer=tokenizer,
            spec_tokens_dict=spec_tokens_dict,
            return_tensors=return_tensors,
        )
        self.inp_prefix = inp_prefix
        self.out_prefix = out_prefix
        self.inp_padding_side = inp_padding_side
        self.out_padding_side = out_padding_side
        self.inp_truncation_side = inp_truncation_side
        self.out_truncation_side = out_truncation_side
        self.inp_max_len = inp_max_len
        self.out_max_len = out_max_len
        self.inp_padding = inp_padding
        self.out_padding = out_padding
        self.inp_truncation = inp_truncation
        self.out_truncation = out_truncation
        self.inp_add_special_tokens = inp_add_special_tokens
        self.out_add_special_tokens = out_add_special_tokens
        self.knowledge_separator = knowledge_separator

    def __call__(self, batch) -> Dict:
        inp = batch.get("turn", None)
        if inp is not None:
            inp = [sample["text"] for sample in inp]
            batch["inp"] = [
                self.tokenize(
                    prefix=self.inp_prefix,
                    text=inp,
                    add_special_tokens=self.inp_add_special_tokens,
                    padding=self.inp_padding,
                    truncation=self.inp_truncation,
                    max_length=self.inp_max_len,
                    padding_side=self.inp_padding_side,
                    truncation_side=self.inp_truncation_side,
                )
            ]
            batch.pop("turn")

        out = batch.get("gk", None)
        if out is not None:
            out = [
                " ".join([self.knowledge_separator + " " + gk for gk in sample])
                for sample in out
            ]
            batch["out"] = [
                self.tokenize(
                    prefix=self.out_prefix,
                    text=out,
                    add_special_tokens=self.out_add_special_tokens,
                    padding=self.out_padding,
                    truncation=self.out_truncation,
                    max_length=self.out_max_len,
                    padding_side=self.out_padding_side,
                    truncation_side=self.out_truncation_side,
                )
            ]
            batch.pop("gk")
        return batch


class KnowledgeGrounGenerationCollator(BaseCollator):
    def __init__(
        self,
        tokenizer,
        knowledge_separator: str,
        spec_tokens_dict: dict,
        inp_prefix: str,
        out_prefix: str = None,
        inp_padding_side: Literal["left", "right"] = "right",
        out_padding_side: Literal["left", "right"] = "right",
        inp_truncation_side: Literal["left", "right"] = "right",
        out_truncation_side: Literal["left", "right"] = "right",
        inp_max_len: int = 64,
        out_max_len: int = 64,
        inp_padding: Literal["max_length", True] = True,
        out_padding: Literal["max_length", True] = True,
        inp_truncation: bool = True,
        out_truncation: bool = True,
        inp_add_special_tokens: bool = True,
        out_add_special_tokens: bool = True,
        return_tensors: Literal["pt"] = "pt",
    ) -> None:
        super().__init__(
            tokenizer=tokenizer,
            spec_tokens_dict=spec_tokens_dict,
            return_tensors=return_tensors,
        )
        self.inp_prefix = inp_prefix
        self.out_prefix = out_prefix
        self.inp_padding_side = inp_padding_side
        self.out_padding_side = out_padding_side
        self.inp_truncation_side = inp_truncation_side
        self.out_truncation_side = out_truncation_side
        self.inp_max_len = inp_max_len
        self.out_max_len = out_max_len
        self.inp_padding = inp_padding
        self.out_padding = out_padding
        self.inp_truncation = inp_truncation
        self.out_truncation = out_truncation
        self.inp_add_special_tokens = inp_add_special_tokens
        self.out_add_special_tokens = out_add_special_tokens
        self.knowledge_separator = knowledge_separator

    def __call__(self, batch) -> Dict:
        inp = batch.get("turn", None)
        if inp is not None:
            inp = [sample["text"] for sample in inp]
            batch["inp"] = [
                self.tokenize(
                    prefix=self.inp_prefix,
                    text=inp,
                    add_special_tokens=self.inp_add_special_tokens,
                    padding=self.inp_padding,
                    truncation=self.inp_truncation,
                    max_length=self.inp_max_len,
                    padding_side=self.inp_padding_side,
                    truncation_side=self.inp_truncation_side,
                )
            ]
            batch.pop("turn")

        out = batch.get("gk", None)
        if out is not None:
            out = [
                " ".join([self.knowledge_separator + " " + gk for gk in sample])
                for sample in out
            ]
            batch["out"] = [
                self.tokenize(
                    prefix=self.out_prefix,
                    text=out,
                    add_special_tokens=self.out_add_special_tokens,
                    padding=self.out_padding,
                    truncation=self.out_truncation,
                    max_length=self.out_max_len,
                    padding_side=self.out_padding_side,
                    truncation_side=self.out_truncation_side,
                )
            ]
            batch.pop("gk")
        return batch
