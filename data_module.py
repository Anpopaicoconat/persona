import json
import random
from typing import *
import os

import random
import json
import pandas as pd
import pytorch_lightning as pl
import torch
import transformers
import datasets as ds


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


class MainCollator:
    def __init__(
        self,
        tokenizer: str,
        spec_tokens: dict,
    ):
        self.tokenizer = tokenizer
        self.spec_tokens = spec_tokens

    def __call__(self, batch, task):
        if task == "current_gk":
            return self.current_gk(batch, task)
        elif task == "next_gk":
            return self.next_gk(batch, task)
        elif task == "next_answer":
            return self.next_answer(batch, task)

    def current_gk(self, batch, task):
        # input
        query = [self.make_msg(turn) for turn in batch["turn"]]
        query = self.tokenizer(
            query,
            padding=True,
            truncation=True,
            add_special_tokens=False,
            max_length=64,
            return_tensors="pt",
        )

        query_prefix = self.spec_tokens["task3"]
        query_prefix = self.tokenizer(
            [query_prefix], add_special_tokens=False, return_tensors="pt"
        )
        query = self.add_prefix(query, query_prefix)
        # output
        candidate = [self.make_gk(gk) for gk in batch["gk"]]
        candidate = self.tokenizer(
            candidate,
            padding=True,
            truncation=True,
            add_special_tokens=False,
            max_length=32,
            return_tensors="pt",
        )
        # for q, c in zip(
        #     self.tokenizer.batch_decode(query["input_ids"]),
        #     self.tokenizer.batch_decode(candidate["input_ids"]),
        # ):
        #     print(q)
        #     print(c)
        #     print()
        # 0 / 0
        return {"task": task, "query": query, "candidate": candidate}

    def next_gk(self, batch, task):
        # query
        query_texts = [self.make_dialog(dialog) for dialog in batch["history"]]
        query = self.tokenizer(
            query_texts,
            padding=True,
            truncation=True,
            add_special_tokens=False,
            max_length=64,
            return_tensors="pt",
        )
        query_prefix = self.spec_tokens["task4q"]
        query_prefix = self.tokenizer(
            [query_prefix], add_special_tokens=False, return_tensors="pt"
        )
        query = self.add_prefix(query, query_prefix)

        # candidate
        candidate = batch["gk"]
        candidate = self.tokenizer(
            candidate,
            padding=True,
            truncation=True,
            add_special_tokens=False,
            max_length=32,
            return_tensors="pt",
        )
        candidate_prefix = self.spec_tokens["task4c"]
        candidate_prefix = self.tokenizer(
            [candidate_prefix], add_special_tokens=False, return_tensors="pt"
        )
        candidate = self.add_prefix(candidate, candidate_prefix)

        # labels
        labels = self.make_labels(query_texts, batch["gk"], batch["all_gks"])

        return {"task": task, "query": query, "candidate": candidate, "labels": labels}

    def next_answer(self, batch, task):
        # query
        query = [self.make_dialog(dialog) for dialog in batch["history"]]
        gk = [self.make_gk(gk) for gk in batch["gk"]]
        query = [q + g for q, g in zip(query, gk)]
        query = self.tokenizer(
            query,
            padding=True,
            truncation=True,
            add_special_tokens=False,
            max_length=64,
            return_tensors="pt",
        )

        query_prefix = self.spec_tokens["task1"]
        query_prefix = self.tokenizer(
            [query_prefix], add_special_tokens=False, return_tensors="pt"
        )
        query = self.add_prefix(query, query_prefix)

        # candidate
        candidate = batch["answer"]
        candidate = [self.make_msg(c) for c in candidate]
        candidate = self.tokenizer(
            candidate,
            padding=True,
            truncation=True,
            add_special_tokens=False,
            max_length=32,
            return_tensors="pt",
        )
        # for q, c in zip(
        #     self.tokenizer.batch_decode(query["input_ids"]),
        #     self.tokenizer.batch_decode(candidate["input_ids"]),
        # ):
        #     print(q)
        #     print(c)
        #     print()
        # 0 / 0
        return {"task": task, "query": query, "candidate": candidate}

    def make_msg(self, msg_dict):
        text = msg_dict["text"]
        gender = msg_dict["gender"]
        if gender == "male":
            gender = self.spec_tokens["male_gender"]
        if gender == "female":
            gender = self.spec_tokens["female_gender"]
        else:
            gender = self.spec_tokens["unknown_gender"]
        return gender + text

    def make_dialog(self, dialog_list):
        c_out = self.spec_tokens["model_answer"]
        for i, turn in enumerate(dialog_list[::-1]):
            if i % 2 == 0:
                P = self.spec_tokens["user_answer"]
            else:
                P = self.spec_tokens["model_answer"]
            c_out = P + self.make_msg(turn) + c_out
        return c_out

    def make_gk(self, gk_list):
        gk_add = self.spec_tokens["model_gk"]
        gk_list = [gk_add + gk for gk in gk_list]
        return "".join(gk_list)

    def add_prefix(self, input, prefix):
        for k in input:
            input[k] = torch.concat(
                [prefix[k].repeat((input[k].size()[0], 1)), input[k]],
                dim=1,
            )
        return input

    def make_labels(self, query, candidate, all_relevant_candidates):
        # TODO: проверку по совпадению
        # targets = torch.eye(query["input_ids"].size()[0], dtype=torch.long)

        labels = []
        for i, q in enumerate(query):
            row = []
            for c in candidate:
                if c in all_relevant_candidates[i]:
                    row.append(1)
                else:
                    row.append(0)
            labels.append(row)
        return torch.tensor(labels)
