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
            self.collator = KnowledgeGrounGenerationCollator(
                tokenizer=tokenizer, spec_tokens_dict=spec_tokens_dict, **collator_conf
            )
        elif task_name in ["knowledge_retrieval"]:
            self.collator = KnowledgeRetrievalCollator(
                tokenizer=tokenizer, spec_tokens_dict=spec_tokens_dict, **collator_conf
            )
        elif task_name in ["knowledge_extraction"]:
            self.collator = KnowledgeExtractionCollator(
                tokenizer=tokenizer, spec_tokens_dict=spec_tokens_dict, **collator_conf
            )
        # elif task_name in ["answer_confirmation"]:
        #     self.collator = KnowledgeExtractionCollator(
        #         tokenizer=tokenizer, spec_tokens_dict=spec_tokens_dict, **collator_conf
        #     )

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
        num_turns: int = -1,
        knowledge_prefix: int = 64,
        knowledge_padding_side: Literal["left", "right"] = "right",
        knowledge_truncation_side: Literal["left", "right"] = "right",
        knowledge_max_len: int = 64,
        knowledge_padding: Literal["max_length", True] = True,
        knowledge_truncation: bool = True,
        knowledge_add_special_tokens: bool = True,
        speakers_seps: Any = [],
        turn_separator: Any = "",
        knowledge_separator: Any = "",
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

        self.knowledge_prefix = knowledge_prefix
        self.knowledge_padding_side = knowledge_padding_side
        self.knowledge_truncation_side = knowledge_truncation_side
        self.knowledge_max_len = knowledge_max_len
        self.knowledge_padding = knowledge_padding
        self.knowledge_truncation = knowledge_truncation
        self.knowledge_add_special_tokens = knowledge_add_special_tokens

        self.speakers_seps = speakers_seps
        self.turn_separator = turn_separator
        self.num_turns = num_turns

    def __call__(self, batch) -> Dict:
        # history
        history_l = []
        history = batch.get("history", None)
        if history is not None:
            if self.num_turns != -1:
                history = [i[-self.num_turns :] for i in history]
            history_new = []
            for sample in history:
                sample_new = []
                for i, turn in enumerate(sample):
                    if self.speakers_seps is not None:
                        turn = self.speakers_seps[i % 2] + turn["text"]
                        sample_new.append(turn)
                history_l.append(len(sample_new))
                sample = self.turn_separator.join(sample_new)
                history_new.append(sample)

            history = self.tokenize(
                prefix=self.inp_prefix,
                text=history_new,
                add_special_tokens=self.inp_add_special_tokens,
                padding=self.inp_padding,
                truncation=self.inp_truncation,
                max_length=self.inp_max_len,
                padding_side=self.inp_padding_side,
                truncation_side=self.inp_truncation_side,
            )
            batch.pop("history")

        # gks
        gks = batch.get("gk", None)
        if gks is not None:
            gks = [
                " ".join([self.knowledge_separator + " " + gk for gk in sample])
                for sample in gks
            ]
            gks = self.tokenize(
                prefix=self.knowledge_prefix,
                text=gks,
                add_special_tokens=self.knowledge_add_special_tokens,
                padding=self.knowledge_padding,
                truncation=self.knowledge_truncation,
                max_length=self.knowledge_max_len,
                padding_side=self.knowledge_padding_side,
                truncation_side=self.knowledge_truncation_side,
            )
            batch.pop("gk")
            # TODO: возможность inp без gk
            inp = {k: torch.concat([history[k], gks[k]], dim=1) for k in history}

            batch["inp"] = inp

        # answer
        answer = batch.get("answer", None)
        if answer is not None:
            answer = [
                self.speakers_seps[i % 2] + turn["text"]
                for turn, i in zip(answer, history_l)
            ]
            answer = self.tokenize(
                prefix=self.out_prefix,
                text=answer,
                add_special_tokens=self.out_add_special_tokens,
                padding=self.out_padding,
                truncation=self.out_truncation,
                max_length=self.out_max_len,
                padding_side=self.out_padding_side,
                truncation_side=self.out_truncation_side,
            )
            batch.pop("answer")
            batch["out"] = answer

        return batch


class KnowledgeRetrievalCollator(BaseCollator):
    def __init__(
        self,
        tokenizer,
        outputs,
        spec_tokens_dict: dict,
        query_prefix: str,
        out_prefix: str = None,
        query_padding_side: Literal["left", "right"] = "right",
        out_padding_side: Literal["left", "right"] = "right",
        query_truncation_side: Literal["left", "right"] = "right",
        out_truncation_side: Literal["left", "right"] = "right",
        query_max_len: int = 64,
        out_max_len: int = 64,
        query_padding: Literal["max_length", True] = True,
        out_padding: Literal["max_length", True] = True,
        query_truncation: bool = True,
        out_truncation: bool = True,
        query_add_special_tokens: bool = True,
        out_add_special_tokens: bool = True,
        return_tensors: Literal["pt"] = "pt",
        candidate_prefix: int = 64,
        candidate_padding_side: Literal["left", "right"] = "right",
        candidate_truncation_side: Literal["left", "right"] = "right",
        candidate_max_len: int = 64,
        candidate_padding: Literal["max_length", True] = True,
        candidate_truncation: bool = True,
        candidate_add_special_tokens: bool = True,
        speakers_seps: Any = [],
        candidate_separator: Any = "",
    ) -> None:
        super().__init__(
            tokenizer=tokenizer,
            spec_tokens_dict=spec_tokens_dict,
            return_tensors=return_tensors,
        )
        self.query_prefix = query_prefix
        self.out_prefix = out_prefix
        self.query_padding_side = query_padding_side
        self.out_padding_side = out_padding_side
        self.query_truncation_side = query_truncation_side
        self.out_truncation_side = out_truncation_side
        self.query_max_len = query_max_len
        self.out_max_len = out_max_len
        self.query_padding = query_padding
        self.out_padding = out_padding
        self.query_truncation = query_truncation
        self.out_truncation = out_truncation
        self.query_add_special_tokens = query_add_special_tokens
        self.out_add_special_tokens = out_add_special_tokens

        self.candidate_prefix = candidate_prefix
        self.candidate_padding_side = candidate_padding_side
        self.candidate_truncation_side = candidate_truncation_side
        self.candidate_max_len = candidate_max_len
        self.candidate_padding = candidate_padding
        self.candidate_truncation = candidate_truncation
        self.candidate_add_special_tokens = candidate_add_special_tokens

        self.speakers_seps = speakers_seps
        self.candidate_separator = candidate_separator
        self.outputs = outputs

    def __call__(self, batch) -> List:
        query_new = []
        candidate_new = []
        for i in batch["turn"]:
            for k in batch["gk"]:
                query_new.append(i)
                candidate_new.append(k)
        batch["turn"] = query_new
        batch["gk"] = candidate_new

        # history
        query = batch.get("turn", None)
        if query is not None:
            query = [sample["text"] for sample in query]
            query = self.tokenize(
                prefix=self.query_prefix,
                text=query,
                add_special_tokens=self.query_add_special_tokens,
                padding=self.query_padding,
                truncation=self.query_truncation,
                max_length=self.query_max_len,
                padding_side=self.query_padding_side,
                truncation_side=self.query_truncation_side,
            )
            batch.pop("turn")

        # gks
        candidate = batch.get("gk", None)
        if candidate is not None:
            # 1 random gk for sample in ep
            candidate = [
                self.candidate_separator + " " + random.choice(sample)
                for sample in candidate
            ]
            candidate = self.tokenize(
                prefix=self.candidate_prefix,
                text=candidate,
                add_special_tokens=self.candidate_add_special_tokens,
                padding=self.candidate_padding,
                truncation=self.candidate_truncation,
                max_length=self.candidate_max_len,
                padding_side=self.candidate_padding_side,
                truncation_side=self.candidate_truncation_side,
            )
            batch.pop("gk")

        inp = {k: torch.concat([query[k], candidate[k]], dim=1) for k in query}
        batch["inp"] = inp

        labels = torch.eye(candidate["input_ids"].size()[0])
        out = [
            self.outputs[1] if label else self.outputs[0] for label in labels.flatten()
        ]
        batch["out"] = self.tokenize(
            prefix=self.out_prefix,
            text=out,
            add_special_tokens=self.out_add_special_tokens,
            padding=self.out_padding,
            truncation=self.out_truncation,
            max_length=self.out_max_len,
            padding_side=self.out_padding_side,
            truncation_side=self.out_truncation_side,
        )

        return batch
