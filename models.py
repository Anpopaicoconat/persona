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

import gc


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
            train_metrics, on_epoch=True, on_step=False, batch_size=self.batch_size
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
            val_metrics, on_epoch=True, on_step=False, batch_size=self.batch_size
        )
        self.log(
            "val_loss",
            val_loss,
            on_step=False,
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

        self.log_dict(metrics, on_epoch=True, on_step=False, batch_size=self.batch_size)
        self.log(
            "val_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=self.batch_size,
        )
        return loss

    def validation_epoch_end(self, outputs):
        self.val_metrics.reset()


class T5_Model(pl.LightningModule):
    def __init__(
        self,
        T5,
        train_answer_dataloader,
        val_answer_dataloader,
        train_gk_dataloader,
        val_gk_dataloader,
        tokenizer,
        scheduler_len,
        num_warmup_steps,
        lr,
    ):
        super().__init__()
        self.T5 = T5

        self.train_answer_dataloader = train_answer_dataloader
        self.val_answer_dataloader = val_answer_dataloader
        self.train_gk_dataloader = train_gk_dataloader
        self.val_gk_dataloader = val_gk_dataloader

        self.tokenizer = tokenizer
        self.scheduler_len = scheduler_len
        self.num_warmup_steps = num_warmup_steps
        self.lr = lr

        # BiEncoder
        self.BiEnc_answer_loss = torch.nn.CrossEntropyLoss()
        self.BiEnc_gk_loss = torch.nn.CrossEntropyLoss()
        BiEnc_metrics = torchmetrics.MetricCollection(
            {
                "BiEnc_r1": torchmetrics.RetrievalRecall(k=1),
                "BiEnc_r5": torchmetrics.RetrievalRecall(k=5),
                "BiEnc_mrr": torchmetrics.RetrievalMRR(),
            }
        )
        self.train_BiEnc_answer_metrics = BiEnc_metrics.clone(prefix="train_answer_")
        self.train_BiEnc_gk_metrics = BiEnc_metrics.clone(prefix="train_gk_")
        self.val_BiEnc_answer_metrics = BiEnc_metrics.clone(prefix="val_answer_")
        self.val_BiEnc_gk_metrics = BiEnc_metrics.clone(prefix="val_gk_")

        # Generative
        self.gen_metrics = torchmetrics.MetricCollection(
            {
                "val_BLEU1": torchmetrics.BLEUScore(n_gram=1),
                "val_BLEU2": torchmetrics.BLEUScore(n_gram=2),
                "val_BLEU4": torchmetrics.BLEUScore(n_gram=4),
            }
        )

        self.save_hyperparameters(ignore=["T5"])

    def train_dataloader(self):
        return {
            "answer_dataloader": self.train_answer_dataloader,
            "gk_dataloader": self.train_gk_dataloader,
        }

    def val_dataloader(self):
        return [
            self.val_answer_dataloader,
            self.val_gk_dataloader,
        ]

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        scheduler = transformers.get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.num_warmup_steps,
            num_training_steps=self.scheduler_len,
        )
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]

    def BiEncode(self, batch):
        """encode query or candidate for bi-encoder with use of encoder part of t5

        Args:
            batch (_type_): _description_

        Returns:
            _type_: _description_
        """
        batch = self.T5(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["input_ids"],
            output_hidden_states=True,
        )
        batch = batch.encoder_last_hidden_state[:, 0, :]
        return batch

    def compute_sim(self, x, y):
        x = x / x.norm(dim=1)[:, None]
        y = y / y.norm(dim=1)[:, None]
        return torch.mm(x, y.transpose(0, 1)) * 10

    def BiEnc_rank(self, batch):
        for k in batch:
            if k == "context":
                query = self.BiEncode(batch[k])
            elif k == "answer":
                candidat = self.BiEncode(batch[k])
                loss_f = self.BiEnc_answer_loss
            elif k == "gk":
                candidat = self.BiEncode(batch[k])
                loss_f = self.BiEnc_gk_loss

        sim = self.compute_sim(query, candidat)
        b_size = query.size()[0]
        labels = torch.zeros((b_size, b_size), dtype=torch.long, device=self.device)
        labels.fill_diagonal_(1)
        loss = loss_f(sim, torch.argmax(labels, 1))
        preds = sim.view(-1)
        targets = labels.view(-1)
        indexes = (
            torch.arange(sim.shape[0]).unsqueeze(1).expand_as(sim).reshape(preds.shape)
        )
        lr = self.trainer.optimizers[0].param_groups[0]["lr"]

        return {
            "loss": loss,
            "preds": preds,
            "targets": targets,
            "indexes": indexes,
            "lr": lr,
            "b_size": b_size,
        }

    def training_step(self, batch, batch_idx):
        BiEnc_answer_out = self.BiEnc_rank(batch["answer_dataloader"])
        BiEnc_gk_out = self.BiEnc_rank(batch["gk_dataloader"])
        loss = BiEnc_gk_out["loss"]
        # (BiEnc_answer_out["loss"] + BiEnc_gk_out["loss"]) / 2

        # log all
        self.log(
            "lr",
            BiEnc_answer_out["lr"],
            on_epoch=False,
            on_step=True,
            prog_bar=True,
            batch_size=BiEnc_answer_out["b_size"],
        )
        self.log(
            "train_loss",
            loss,
            on_epoch=True,
            on_step=True,
            prog_bar=True,
            batch_size=BiEnc_answer_out["b_size"],
        )
        # log answer
        self.log(
            "train_BiEnc_answer_loss",
            BiEnc_answer_out["loss"],
            on_epoch=True,
            on_step=True,
            prog_bar=True,
            batch_size=BiEnc_answer_out["b_size"],
        )
        m = self.train_BiEnc_answer_metrics(
            BiEnc_answer_out["preds"],
            BiEnc_answer_out["targets"],
            indexes=BiEnc_answer_out["indexes"],
        )
        self.log_dict(
            m,
            on_epoch=True,
            on_step=True,
            batch_size=BiEnc_answer_out["b_size"],
        )
        # log gk
        self.log(
            "train_BiEnc_gk_loss",
            BiEnc_gk_out["loss"],
            on_epoch=True,
            on_step=False,
            prog_bar=True,
            batch_size=BiEnc_gk_out["b_size"],
        )
        m = self.train_BiEnc_gk_metrics(
            BiEnc_gk_out["preds"],
            BiEnc_gk_out["targets"],
            indexes=BiEnc_gk_out["indexes"],
        )
        self.log_dict(
            m,
            on_epoch=True,
            on_step=False,
            batch_size=BiEnc_gk_out["b_size"],
        )
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx):
        if dataloader_idx == 0:
            BiEnc_answer_out = self.BiEnc_rank(batch)
            # log answer
            self.log(
                "val_BiEnc_answer_loss",
                BiEnc_answer_out["loss"],
                on_epoch=True,
                on_step=True,
                prog_bar=True,
                batch_size=BiEnc_answer_out["b_size"],
                add_dataloader_idx=False,
            )
            m = self.val_BiEnc_answer_metrics(
                BiEnc_answer_out["preds"],
                BiEnc_answer_out["targets"],
                indexes=BiEnc_answer_out["indexes"],
                add_dataloader_idx=False,
            )
            self.log_dict(
                m,
                on_epoch=True,
                on_step=True,
                batch_size=BiEnc_answer_out["b_size"],
                add_dataloader_idx=False,
            )
        elif dataloader_idx == 1:
            BiEnc_gk_out = self.BiEnc_rank(batch)
            # log gk
            self.log(
                "val_BiEnc_gk_loss",
                BiEnc_gk_out["loss"],
                on_epoch=True,
                on_step=True,
                prog_bar=True,
                batch_size=BiEnc_gk_out["b_size"],
                add_dataloader_idx=False,
            )
            m = self.val_BiEnc_gk_metrics(
                BiEnc_gk_out["preds"],
                BiEnc_gk_out["targets"],
                indexes=BiEnc_gk_out["indexes"],
            )
            self.log_dict(
                m,
                on_epoch=True,
                on_step=True,
                batch_size=BiEnc_gk_out["b_size"],
                add_dataloader_idx=False,
            )

    def train_epoch_end(self, outputs):
        self.train_BiEnc_answer_metrics.reset()
        self.train_BiEnc_gk_metrics.reset()

    def validation_epoch_end(self, outputs):
        self.val_BiEnc_answer_metrics.reset()
        self.val_BiEnc_gk_metrics.reset()
