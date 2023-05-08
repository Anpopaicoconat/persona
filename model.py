from typing import *

import pytorch_lightning as pl
import torch
import torchmetrics
import transformers
import pandas as pd
import numpy as np
import os
from data import MultiDataModule
from utils import *

import requests


class MultitaskModel(pl.LightningModule):
    def __init__(
        self,
        model_name_or_path: str,
        data_dir: str,
        spec_tokens_dict: Dict,
        tasks: Dict,
        seed: int = 42,
        lr: float = 5e-05,
        num_warmup_steps: int = 100,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.transformer = transformers.T5ForConditionalGeneration.from_pretrained(
            self.hparams.model_name_or_path
        )
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(model_name_or_path)
        # init tasks
        self.metrics_dict = {}
        task_list = []
        for task_name in tasks:
            task = TaskConfig(
                task_name=task_name,
                tokenizer=self.tokenizer,
                spec_tokens_dict=spec_tokens_dict,
                **tasks[task_name],
            )
            task_list.append(task)
            # init metrics
            self.metrics_dict[task.task_name] = {}
            for ds in task.datasets:
                self.metrics_dict[task.task_name][ds] = {}
                if task.task_name in [
                    "knowledge_ground_generation",
                    "knowledge_extraction",
                ]:
                    metrics = torchmetrics.MetricCollection(
                        {
                            f"{task.task_name}_{ds}_BLEU1": torchmetrics.BLEUScore(
                                n_gram=1
                            ),
                            f"{task.task_name}_{ds}_BLEU2": torchmetrics.BLEUScore(
                                n_gram=2
                            ),
                        }
                    )
                elif task.task_name in ["knowledge_retrieval"]:
                    metrics = torchmetrics.MetricCollection(
                        {
                            f"{task.task_name}_{ds}_Recall": torchmetrics.Recall(
                                task="multiclass",
                                num_classes=len(task.collator.outputs),
                            ),
                        }
                    )
                self.metrics_dict[task.task_name][ds]["train"] = metrics.clone(
                    prefix="train_"
                )
                self.metrics_dict[task.task_name][ds]["val"] = metrics.clone(
                    prefix="val_"
                )
        # init data module
        self.data_module = MultiDataModule(
            tasks=task_list,
            tokenizer=self.tokenizer,
            data_dir=self.hparams.data_dir,
            spec_tokens=self.hparams.spec_tokens_dict,
            seed=self.hparams.seed,
        )
        self.transformer.resize_token_embeddings(len(self.tokenizer))

    def seq2seq(self, batch, meta):
        loss = self.transformer(**batch["inp"], labels=batch["out"]["input_ids"]).loss
        out = self.transformer.generate(
            **batch["inp"], max_length=batch["out"]["input_ids"].size()[-1] + 1
        )
        pred = self.data_module.multi_collator.label_decode(batch=out, **meta)
        target = self.data_module.multi_collator.label_decode(
            batch=batch["out"]["input_ids"], **meta
        )
        return loss, pred, target

    def forward(self, batch, meta):
        if meta["task_name"] in [
            "knowledge_ground_generation",
            "knowledge_retrieval",
            "knowledge_extraction",
        ]:
            loss, pred, target = self.seq2seq(batch, meta)

        return loss, pred, target

    def training_step(self, batch: dict, batch_idx):
        loss, pred, target = self(batch[batch["type"]["task_name"]], batch["type"])
        metrics = self.metrics_dict[batch["type"]["task_name"]][
            batch["type"]["ds_name"]
        ]["train"](pred, target)
        # Log
        self.log("train_loss", loss, sync_dist=True)
        self.log_dict(metrics, sync_dist=True)

        return loss

    def training_epoch_end(self, outputs):
        for task_name in self.metrics_dict:
            for ds_name in self.metrics_dict[task_name]:
                print(self.metrics_dict[task_name][ds_name]["train"].compute())
                self.metrics_dict[task_name][ds_name]["train"].reset()

    def validation_step(self, batch: dict, batch_idx):
        return 1

    def validation_epoch_end(self, outputs):
        for task_name in self.metrics_dict:
            for ds_name in self.metrics_dict[task_name]:
                self.metrics_dict[task_name][ds_name]["val"].reset()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)
        scheduler = transformers.get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.hparams.num_warmup_steps,
            num_training_steps=self.trainer.estimated_stepping_batches,
        )
        return [optimizer], [
            {"scheduler": scheduler, "name": "cosine_scheduler", "interval": "step"}
        ]

    def retrieve(
        self, batch: Dict, task_name: str, ds_name: str = "encode", max_length: int = 32
    ):
        self.eval()
        batch = self.data_module.multi_collator(task_name, ds_name, batch)
        out = self.transformer.generate(**batch["inp"], max_length=max_length)
        pred = self.data_module.multi_collator.label_decode(
            batch=out, task_name=task_name, ds_name=ds_name
        )
        return pred
