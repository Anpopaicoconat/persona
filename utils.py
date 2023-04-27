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
            self.collator = PairInBatchLabelCollator(
                tokenizer=tokenizer, spec_tokens_dict=spec_tokens_dict, **collator_conf
            )
        elif task_name in ["knowledge_retrieval"]:
            self.collator = DialogCollator(
                tokenizer=tokenizer, spec_tokens_dict=spec_tokens_dict, **collator_conf
            )
        elif task_name in ["knowledge_extraction"]:
            self.collator = DialogCollator(
                tokenizer=tokenizer, spec_tokens_dict=spec_tokens_dict, **collator_conf
            )
        elif task_name in ["answer_confirmation"]:
            self.collator = DialogCollator(
                tokenizer=tokenizer, spec_tokens_dict=spec_tokens_dict, **collator_conf
            )

        self.datasets = datasets


class TolokaDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        datasets: List[str],
        tokenizer: str,
        spec_tokens: dict,
        train_batch_size: int,
        val_batch_size: int,
        test_batch_size: int,
        seed: int = 42,
    ):
        super().__init__()
        self.save_hyperparameters()
        datasets_path = [
            os.path.join(self.hparams.data_dir, dataset_name)
            for dataset_name in self.hparams.datasets
        ]
        datasets_instanse = [ds.load_from_disk(path) for path in datasets_path]
        self.datasets = {k: v for k, v in zip(self.hparams.datasets, datasets_instanse)}

        self.tokenizer = tokenizer
        self.collator = MainCollator(tokenizer, spec_tokens)

    def train_dataloader(self):
        try:
            ep = self.trainer.current_epoch
        except:
            ep = 0
        # shuffle train split
        datasets = {
            dataset_name: self.datasets[dataset_name]["train"].shuffle(
                seed=self.hparams.seed + ep
            )
            for dataset_name in self.datasets
        }
        # make batch
        datasets = [
            datasets[task].map(
                lambda batch, task: {
                    task: [self.collator(batch, task)],
                    "task": [task],
                },
                batched=True,
                batch_size=self.hparams.train_batch_size,
                remove_columns=datasets[task].column_names,
                fn_kwargs={"task": task},
                drop_last_batch=True,
                num_proc=1,
            )
            for task in datasets
        ]
        ds_num_batch = [len(ds) for ds in datasets]
        ds_prob = [size / sum(ds_num_batch) for size in ds_num_batch]
        train_dataloader = ds.interleave_datasets(datasets, probabilities=ds_prob)
        return train_dataloader.with_format("pytorch")

    def val_dataloader(self):
        try:
            ep = self.trainer.current_epoch
        except:
            ep = 0
        # shuffle train split
        datasets = {
            dataset_name: self.datasets[dataset_name]["val"].shuffle(
                seed=self.hparams.seed + ep
            )
            for dataset_name in self.datasets
        }
        # make batch
        datasets = [
            datasets[task].map(
                lambda batch, task: {
                    task: [self.collator(batch, task)],
                    "task": [task],
                },
                batched=True,
                batch_size=self.hparams.train_batch_size,
                remove_columns=datasets[task].column_names,
                fn_kwargs={"task": task},
                drop_last_batch=True,
                num_proc=1,
            )
            for task in datasets
        ]
        ds_num_batch = [len(ds) for ds in datasets]
        ds_prob = [size / sum(ds_num_batch) for size in ds_num_batch]
        train_dataloader = ds.interleave_datasets(datasets, probabilities=ds_prob)
        return train_dataloader.with_format("pytorch")

    def test_dataloader(self):
        try:
            ep = self.trainer.current_epoch
        except:
            ep = 0


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


class PairCollator(BaseCollator):
    def __init__(
        self,
        tokenizer,
        spec_tokens_dict: dict,
        query_prefix: str,
        candidate_prefix: str = None,
        query_padding_side: Literal["left", "right"] = "right",
        candidate_padding_side: Literal["left", "right"] = "right",
        query_truncation_side: Literal["left", "right"] = "right",
        candidate_truncation_side: Literal["left", "right"] = "right",
        query_max_len: int = 64,
        candidate_max_len: int = 64,
        query_padding: Literal["max_length", True] = True,
        candidate_padding: Literal["max_length", True] = True,
        query_truncation: bool = True,
        candidate_truncation: bool = True,
        query_add_special_tokens: bool = True,
        candidate_add_special_tokens: bool = True,
        return_tensors: Literal["pt"] = "pt",
    ) -> None:
        super().__init__(
            tokenizer=tokenizer,
            spec_tokens_dict=spec_tokens_dict,
            return_tensors=return_tensors,
        )
        self.query_prefix = query_prefix
        self.candidate_prefix = candidate_prefix
        self.query_padding_side = query_padding_side
        self.candidate_padding_side = candidate_padding_side
        self.query_truncation_side = query_truncation_side
        self.candidate_truncation_side = candidate_truncation_side
        self.query_max_len = query_max_len
        self.candidate_max_len = candidate_max_len
        self.query_padding = query_padding
        self.candidate_padding = candidate_padding
        self.query_truncation = query_truncation
        self.candidate_truncation = candidate_truncation
        self.query_add_special_tokens = query_add_special_tokens
        self.candidate_add_special_tokens = candidate_add_special_tokens

    def __call__(self, batch) -> Dict:
        # query
        query = batch.get("query", None)
        if query is not None:
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

        # candidate
        candidate = batch.get("candidate", None)
        if candidate is not None:
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

        return {"query": query, "candidate": candidate}


class PairInBatchLabelCollator(PairCollator):
    def __call__(self, batch) -> Dict:
        labels = batch.get("labels", None)
        score = batch.get("score", None)
        if labels is None and score is None:
            batch = super().__call__(batch)
            query = batch["query"]
            candidate = batch["candidate"]
            # in batch labels
            if (candidate is not None) and (query is not None):
                # calculate labels
                q = query["input_ids"]
                c = candidate["input_ids"]

                # приводим запросы и кандидаты к одинаковой длине для их сравнения
                c_pad = q.size()[-1] - c.size()[-1]
                if c_pad > 0:
                    c = torch.nn.functional.pad(
                        c,
                        (0, c_pad),
                        mode="constant",
                        value=self.tokenizer.pad_token_id,
                    )
                q_pad = c.size()[-1] - q.size()[-1]
                if q_pad > 0:
                    q = torch.nn.functional.pad(
                        q,
                        (0, q_pad),
                        mode="constant",
                        value=self.tokenizer.pad_token_id,
                    )

                # выставляем 1 для всех одинаковых кандидатов для каждого запроса
                labels_q1 = torch.all(c[:, None, :] == c, dim=-1)
                # выставляем 1 для всех кандидатов одинаковых запросу
                labels_q2 = torch.all(c[:, None, :] == q, dim=-1)
                # объединяем первые 2 матрицы
                labels_q = torch.logical_or(labels_q1, labels_q2)

                # аналогично первым 3 операциям, но наоборот
                labels_c1 = torch.all(q[:, None, :] == q, dim=-1)
                labels_c2 = torch.all(q[:, None, :] == c, dim=-1)
                labels_c = torch.logical_or(labels_c1, labels_c2).transpose(1, 0)

                # ищем одинаковых кандидатов
                mask = labels_q[:, :, None]
                # для одинаковых кандидатов объединяем 1 для всех их запросов
                masked = labels_c * mask
                # если хотя бы 1 из одинаковых кандидатов подходил запрос он будет подходить всем
                labels = torch.any(masked, dim=1)
                return {"query": query, "candidate": candidate, "labels": labels}
            else:
                return batch
        else:
            if labels is not None:
                labels = batch.pop("labels")
            elif score is not None:
                score = batch.pop("score")
            batch = super().__call__(batch)
            if labels is not None:
                batch["labels"] = torch.tensor(labels)
            elif score is not None:
                batch["score"] = torch.tensor(score)
            return batch


class DialogCollator(PairInBatchLabelCollator):
    def __init__(
        self,
        tokenizer,
        spec_tokens_dict: dict,
        query_prefix: str,
        candidate_prefix: str = None,
        turn_separator: str = "",
        speakers_seps: List = [],
        query_padding_side: Literal["left", "right"] = "right",
        candidate_padding_side: Literal["left", "right"] = "right",
        query_truncation_side: Literal["left", "right"] = "right",
        candidate_truncation_side: Literal["left", "right"] = "right",
        query_max_len: int = 64,
        candidate_max_len: int = 64,
        query_padding: Literal["max_length", True] = True,
        candidate_padding: Literal["max_length", True] = True,
        query_truncation: bool = True,
        candidate_truncation: bool = True,
        query_add_special_tokens: bool = True,
        candidate_add_special_tokens: bool = True,
        return_tensors: Literal["pt"] = "pt",
    ) -> None:
        super().__init__(
            tokenizer=tokenizer,
            spec_tokens_dict=spec_tokens_dict,
            return_tensors=return_tensors,
            query_prefix=query_prefix,
            candidate_prefix=candidate_prefix,
            query_padding_side=query_padding_side,
            candidate_padding_side=candidate_padding_side,
            query_truncation_side=query_truncation_side,
            candidate_truncation_side=candidate_truncation_side,
            query_max_len=query_max_len,
            candidate_max_len=candidate_max_len,
            query_padding=query_padding,
            candidate_padding=candidate_padding,
            query_truncation=query_truncation,
            candidate_truncation=candidate_truncation,
            query_add_special_tokens=query_add_special_tokens,
            candidate_add_special_tokens=candidate_add_special_tokens,
        )
        self.turn_separator = turn_separator
        self.speakers_seps = speakers_seps

    def __call__(self, batch) -> Dict:
        query_new = []
        for sample in batch["query"]:
            new_sample = []
            for turn in sample:
                new_sample.append(self.speakers_seps[turn["speaker"]] + turn["text"])
            query_new.append(self.turn_separator.join(new_sample))
        batch["query"] = query_new
        batch["candidate"] = [sample["text"] for sample in batch["candidate"]]
        return super().__call__(batch)


class MultiCollator:
    def __init__(self, collators_dict):
        self.collators_dict = collators_dict

    def __call__(self, task_name, ds_name, batch) -> Dict:
        return {
            "batch": self.collators_dict[task_name](batch),
            "type": {
                "task": task_name,
                "source": ds_name,
            },
        }
