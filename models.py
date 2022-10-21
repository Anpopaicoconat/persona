from cProfile import label
import os
import argparse
import json
import random

import torch
import pytorch_lightning as pl
import torchmetrics
import transformers
from utils import aggregate_encoder_output, sim_func


class BERT_RetrievalModel(pl.LightningModule):
    def __init__(
        self,
        context_BERT,
        candidat_BERT,
        batch_size,
        scheduler_len,
        num_warmup_steps,
        lr,
        aggregation_mod,
        sim_mod,
        tokenizer,
        collator,
        base_config,
    ):
        super().__init__()
        self.context_BERT = context_BERT
        self.candidat_BERT = candidat_BERT
        self.batch_size = batch_size
        self.scheduler_len = scheduler_len
        self.num_warmup_steps = num_warmup_steps
        self.loss = torch.nn.CrossEntropyLoss()
        self.lr = lr
        self.aggregation_mod = aggregation_mod
        self.sim_mod = sim_mod
        self.tokenizer = tokenizer
        self.collator = collator
        self.base_config = base_config
        self.train_metrics = torchmetrics.MetricCollection(
            {
                "train_r1": torchmetrics.RetrievalRecall(k=1),
                "train_r5": torchmetrics.RetrievalRecall(k=5),
                "train_mrr": torchmetrics.RetrievalMRR(),
            }
        )
        self.val_metrics = torchmetrics.MetricCollection(
            {
                "val_r1": torchmetrics.RetrievalRecall(k=1),
                "val_r5": torchmetrics.RetrievalRecall(k=5),
                "val_mrr": torchmetrics.RetrievalMRR(),
            }
        )
        self.save_hyperparameters()

    def training_step(self, batch, batch_idx):
        context = batch["context"]
        # candidate = batch["candidate"]
        candidate = batch["gk"]
        # persona = batch["persona"]
        b_size = context["input_ids"].size()[0]
        # labels = torch.range(0, candidate['input_ids'].size()[0]-1, dtype=torch.long).to(self.device)
        labels = torch.zeros((b_size, b_size), dtype=torch.long).to(self.device)
        labels.fill_diagonal_(1)
        logits = self(context, candidate, torch.argmax(labels, 1))
        loss = self.loss(logits, torch.argmax(labels, 1))
        preds = logits.view(-1)

        targets = labels.view(-1)
        indexes = (
            torch.arange(logits.shape[0])
            .unsqueeze(1)
            .expand_as(logits)
            .reshape(preds.shape)
        )
        train_metrics = self.train_metrics(preds, targets, indexes=indexes)
        self.log_dict(
            train_metrics, on_epoch=True, on_step=True, batch_size=self.batch_size
        )
        self.log(
            "lr",
            self.trainer.optimizers[0].param_groups[0]["lr"],
            on_step=True,
            prog_bar=True,
            batch_size=self.batch_size,
        )
        self.log(
            "train_loss",
            loss,
            on_step=True,
            prog_bar=True,
            batch_size=self.batch_size,
        )

        return loss

    def training_epoch_end(self, outputs):
        self.train_metrics.reset()

    def validation_step(self, val_batch, batch_idx):
        context = val_batch["context"]
        # candidate = val_batch["candidate"]
        candidate = val_batch["gk"]
        # persona = val_batch["persona"]
        b_size = context["input_ids"].size()[0]
        labels = torch.zeros((b_size, b_size), dtype=torch.long).to(self.device)
        labels.fill_diagonal_(1)
        logits = self(context, candidate, torch.argmax(labels, 1))
        val_loss = self.loss(logits, torch.argmax(labels, 1))
        preds = logits.view(-1)
        targets = labels.view(-1)
        indexes = (
            torch.arange(logits.shape[0])
            .unsqueeze(1)
            .expand_as(logits)
            .reshape(preds.shape)
        )
        val_metrics = self.val_metrics(preds, targets, indexes=indexes)
        self.log_dict(
            val_metrics, on_epoch=True, on_step=True, batch_size=self.batch_size
        )
        self.log(
            "val_loss",
            val_loss,
            on_step=True,
            prog_bar=True,
            batch_size=self.batch_size,
        )

        return val_metrics, val_loss

    def validation_epoch_end(self, outputs):
        self.val_metrics.reset()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        scheduler = transformers.get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.num_warmup_steps,
            num_training_steps=self.scheduler_len,
        )
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]

    def forward(self, context, candidat, labels):
        context_vec = self.encode_context(context)
        candidat_vec = self.encode_candidats(candidat)
        distance = self.compute_sim(context_vec, candidat_vec)
        return distance

    def encode_candidats(self, candidat):
        candidat_vec = self.candidat_BERT(**candidat)
        candidat_vec = aggregate_encoder_output(candidat_vec, mod=self.aggregation_mod)
        return candidat_vec

    def encode_context(self, context):
        context_vec = self.context_BERT(**context)
        context_vec = aggregate_encoder_output(context_vec, mod=self.aggregation_mod)
        return context_vec

    def compute_sim(self, context_vec, candidat_vec):
        distance = sim_func(context_vec, candidat_vec, self.sim_mod)
        return distance


class GPT_GenerativeModel(pl.LightningModule):
    def __init__(
        self,
        GPT,
        tokenizer,
        batch_size,
        scheduler_len,
        num_warmup_steps,
        lr,
        max_len,
        collator,
        base_config,
    ):
        super().__init__()
        self.GPT = GPT
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.scheduler_len = scheduler_len
        self.num_warmup_steps = num_warmup_steps
        self.lr = lr
        self.collator = collator
        self.base_config = base_config
        self.max_len = max_len
        self.train_metrics = torchmetrics.MetricCollection({})
        self.val_metrics = torchmetrics.MetricCollection(
            {
                "val_BLEU1": torchmetrics.BLEUScore(n_gram=1),
                "val_BLEU2": torchmetrics.BLEUScore(n_gram=2),
                "val_BLEU4": torchmetrics.BLEUScore(n_gram=4),
                "val_BLEU6": torchmetrics.BLEUScore(n_gram=6),
            }
        )
        self.save_hyperparameters()

    def forward(self, input):
        out = self.GPT(input["input_ids"], labels=input["input_ids"])
        return out

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        scheduler = transformers.get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.num_warmup_steps,
            num_training_steps=self.scheduler_len,
        )
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]

    def training_step(self, batch, batch_idx):
        out = self(batch)
        loss = out.loss

        self.log(
            "lr",
            self.trainer.optimizers[0].param_groups[0]["lr"],
            on_step=True,
            prog_bar=True,
            batch_size=self.batch_size,
        )
        self.log(
            "train_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            batch_size=self.batch_size,
        )
        return loss

    def training_epoch_end(self, outputs):
        self.train_metrics.reset()

    def validation_step(self, batch, batch_idx):
        target_full_ids = batch[0]["input_ids"]
        split = batch[1]
        if split == 0:
            split += 1
        target_context_ids = target_full_ids[:, :split]
        target_candidate_ids = target_full_ids[:, split:]

        output_full_ids = self.GPT.generate(target_context_ids, max_length=self.max_len)
        output_candidate_ids = output_full_ids[:, split:]
        loss = self.GPT(target_full_ids, labels=target_full_ids).loss

        target_candidate = self.tokenizer.batch_decode(target_candidate_ids)
        output_candidate = self.tokenizer.batch_decode(output_candidate_ids)

        metrics = self.val_metrics(output_candidate, [target_candidate])

        self.log_dict(metrics, on_epoch=True, on_step=True, batch_size=self.batch_size)
        self.log(
            "val_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            batch_size=self.batch_size,
        )
        return loss

    def validation_epoch_end(self, outputs):
        self.val_metrics.reset()


class T5_GenerativeModel(pl.LightningModule):
    def __init__(
        self,
        GPT,
        tokenizer,
        batch_size,
        scheduler_len,
        num_warmup_steps,
        lr,
        max_len,
    ):
        super().__init__()
        self.GPT = GPT
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.scheduler_len = scheduler_len
        self.num_warmup_steps = num_warmup_steps
        self.lr = lr
        self.max_len = max_len
        self.train_metrics = torchmetrics.MetricCollection({})
        self.val_metrics = torchmetrics.MetricCollection(
            {
                "val_BLEU1": torchmetrics.BLEUScore(n_gram=1),
                "val_BLEU2": torchmetrics.BLEUScore(n_gram=2),
                "val_BLEU4": torchmetrics.BLEUScore(n_gram=4),
                "val_BLEU6": torchmetrics.BLEUScore(n_gram=6),
            }
        )
        self.save_hyperparameters()

    def forward(self, input):
        out = self.GPT(input["input_ids"], labels=input["output_ids"])
        return out

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        scheduler = transformers.get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.num_warmup_steps,
            num_training_steps=self.scheduler_len,
        )
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]

    def training_step(self, batch, batch_idx):
        out = self(batch)
        loss = out.loss

        self.log(
            "lr",
            self.trainer.optimizers[0].param_groups[0]["lr"],
            on_step=True,
            prog_bar=True,
            batch_size=self.batch_size,
        )
        self.log(
            "train_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            batch_size=self.batch_size,
        )
        return loss

    def training_epoch_end(self, outputs):
        self.train_metrics.reset()

    def validation_step(self, batch, batch_idx):
        target_full_ids = batch[0]["input_ids"]
        split = batch[1]
        if split == 0:
            split += 1
        target_context_ids = target_full_ids[:, :split]
        target_candidate_ids = target_full_ids[:, split:]

        output_full_ids = self.GPT.generate(target_context_ids, max_length=self.max_len)
        output_candidate_ids = output_full_ids[:, split:]
        loss = self.GPT(target_full_ids, labels=target_full_ids).loss

        target_candidate = self.tokenizer.batch_decode(target_candidate_ids)
        output_candidate = self.tokenizer.batch_decode(output_candidate_ids)

        metrics = self.val_metrics(output_candidate, [target_candidate])

        self.log_dict(metrics, on_epoch=True, on_step=True, batch_size=self.batch_size)
        self.log(
            "val_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            batch_size=self.batch_size,
        )
        return loss

    def validation_epoch_end(self, outputs):
        self.val_metrics.reset()
