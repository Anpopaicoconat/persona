from typing import *

import pytorch_lightning as pl
import torch
import torchmetrics
import transformers
import pandas as pd
import os


def parse_recursive_dict(inp_dict, tokens=None):
    tokens = tokens or []
    for k in inp_dict:
        if isinstance(inp_dict[k], dict):
            tokens = parse_recursive_dict(inp_dict[k], tokens=tokens)
        else:
            tokens.append(inp_dict[k])
    return tokens


class T5MultiTask(pl.LightningModule):
    def __init__(
        self,
        model,
        datamodule,
        lr: float = 5e-5,
        num_warmup_steps: int = 100,
        pooling: Literal["mean", "cls"] = "mean",
        distance: Literal["cosine", "dot_product"] = "cosine",
        scale: int = 20,
        train_batch_size: int = 64,
        val_batch_size: int = 64,
        test_batch_size: int = 64,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.transformer = model
        self.datamodule = datamodule
        self.transformer.resize_token_embeddings(len(self.datamodule.tokenizer))

        # loss
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss()

        # metrics
        # next answer
        next_answer_metrics = torchmetrics.MetricCollection(
            {
                "next_answer_BLEU1": torchmetrics.BLEUScore(n_gram=1),
                "next_answer_BLEU2": torchmetrics.BLEUScore(n_gram=2),
            }
        )
        self.train_next_answer_metrics = next_answer_metrics.clone(prefix="train_")
        self.val_next_answer_metrics = next_answer_metrics.clone(prefix="val_")
        # current gk
        current_gk_metrics = torchmetrics.MetricCollection(
            {
                "current_gk_BLEU1": torchmetrics.BLEUScore(n_gram=1),
                "current_gk_BLEU2": torchmetrics.BLEUScore(n_gram=2),
            }
        )
        self.train_current_gk_metrics = current_gk_metrics.clone(prefix="train_")
        self.val_current_gk_metrics = current_gk_metrics.clone(prefix="val_")
        # next gk
        # TODO: add MetricCollection for next gk: r1 r5 mrr

    def get_embedding(self, inputs):
        model_output = self.transformer(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            labels=inputs["input_ids"],
            output_hidden_states=True,
        )
        if self.hparams.pooling == "mean":
            embeddings = self.mean_pooling(model_output, inputs["attention_mask"])
        elif self.hparams.pooling == "cls":
            embeddings = self.cls_pooling(model_output)
        if self.hparams.distance == "cosine":
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

        return embeddings

    def training_step(self, batch: dict, batch_idx):
        task = batch["task"]
        batch = batch[task]
        if task == "next_gk" and False:
            # Compute embeddings
            query = self.get_embedding(batch["query"])
            candidate = self.get_embedding(batch["candidate"])
            labels = torch.argmax(batch["labels"], dim=-1)
            # Compute similarity scores
            scores = torch.mm(query, candidate.transpose(0, 1)) * self.hparams.scale
            # Symmetric loss as in CLIP
            loss = (
                self.cross_entropy_loss(scores, labels)
                + self.cross_entropy_loss(scores.transpose(0, 1), labels)
            ) / 2
            # metrics
            preds = scores.view(-1).cpu()
            targets = batch["labels"].reshape(preds.shape)
            indexes = (
                torch.arange(scores.shape[0])
                .unsqueeze(1)
                .expand_as(scores)
                .reshape(preds.shape)
            )
            # Log
            self.log("train_loss_next_gk", loss.item(), sync_dist=True)
            metrics = self.train_next_gk_metrics[task](preds, targets, indexes)
            self.log_dict(metrics, sync_dist=True)

        # current gk
        elif task == "current_gk":
            model_output = self.transformer(
                input_ids=batch["query"]["input_ids"],
                attention_mask=batch["query"]["attention_mask"],
                labels=batch["candidate"]["input_ids"],
            )
            # loss
            loss = model_output.loss
            # metrics
            model_output = self.transformer.generate(
                input_ids=batch["query"]["input_ids"]
            )
            target_candidate = self.datamodule.tokenizer.batch_decode(
                batch["candidate"]["input_ids"], skip_special_tokens=True
            )
            target_candidate = [[i] for i in target_candidate]
            output_candidate = self.datamodule.tokenizer.batch_decode(
                model_output, skip_special_tokens=True
            )
            metrics = self.train_current_gk_metrics(output_candidate, target_candidate)
            self.log_dict(metrics, sync_dist=True)

        # next answer
        elif task == "next_answer":
            model_output = self.transformer(
                input_ids=batch["query"]["input_ids"],
                attention_mask=batch["query"]["attention_mask"],
                labels=batch["candidate"]["input_ids"],
            )
            # loss
            loss = model_output.loss
            # metrics
            model_output = self.transformer.generate(
                input_ids=batch["query"]["input_ids"],
            )
            target_candidate = self.datamodule.tokenizer.batch_decode(
                batch["candidate"]["input_ids"], skip_special_tokens=True
            )
            target_candidate = [[i] for i in target_candidate]
            output_candidate = self.datamodule.tokenizer.batch_decode(
                model_output, skip_special_tokens=True
            )
            metrics = self.train_next_answer_metrics(output_candidate, target_candidate)
            self.log_dict(metrics, sync_dist=True)

        return loss

    def validation_step(self, batch: dict, batch_idx):
        task = batch["task"]
        batch = batch[task]
        if task == "next_gk" and False:
            # Compute embeddings
            query = self.get_embedding(batch["query"])
            candidate = self.get_embedding(batch["candidate"])
            labels = torch.argmax(batch["labels"], dim=-1)
            # Compute similarity scores
            scores = torch.mm(query, candidate.transpose(0, 1)) * self.hparams.scale
            # Symmetric loss as in CLIP
            # loss = (
            #     self.cross_entropy_loss(scores, labels)
            #     + self.cross_entropy_loss(scores.transpose(0, 1), labels)
            # ) / 2
            loss = self.triplet_loss(batch["labels"], scores)
            # metrics
            preds = scores.view(-1).cpu()
            targets = batch["labels"].reshape(preds.shape)
            indexes = (
                torch.arange(scores.shape[0])
                .unsqueeze(1)
                .expand_as(scores)
                .reshape(preds.shape)
            )
            # Log
            self.log("val_next_gk_loss", loss.item(), sync_dist=True)
            metrics = self.val_next_gk_metrics[task](preds, targets, indexes)
            self.log_dict(metrics, sync_dist=True)

        # current gk
        elif task == "current_gk":
            model_output = self.transformer(
                input_ids=batch["query"]["input_ids"],
                attention_mask=batch["query"]["attention_mask"],
                labels=batch["candidate"]["input_ids"],
            )
            # loss
            loss = model_output.loss
            # metrics
            model_output = self.transformer.generate(
                input_ids=batch["query"]["input_ids"],
            )
            target_candidate = self.datamodule.tokenizer.batch_decode(
                batch["candidate"]["input_ids"], skip_special_tokens=True
            )
            target_candidate = [[i] for i in target_candidate]
            output_candidate = self.datamodule.tokenizer.batch_decode(
                model_output, skip_special_tokens=True
            )
            metrics = self.val_current_gk_metrics(output_candidate, target_candidate)
            self.log_dict(metrics, sync_dist=True)

        # next answer
        elif task == "next_answer":
            model_output = self.transformer(
                input_ids=batch["query"]["input_ids"],
                attention_mask=batch["query"]["attention_mask"],
                labels=batch["candidate"]["input_ids"],
            )
            # loss
            loss = model_output.loss
            # metrics
            model_output = self.transformer.generate(
                input_ids=batch["query"]["input_ids"],
            )
            target_candidate = self.datamodule.tokenizer.batch_decode(
                batch["candidate"]["input_ids"], skip_special_tokens=True
            )
            target_candidate = [[i] for i in target_candidate]
            output_candidate = self.datamodule.tokenizer.batch_decode(
                model_output, skip_special_tokens=True
            )
            metrics = self.val_next_answer_metrics(output_candidate, target_candidate)
            self.log_dict(metrics, sync_dist=True)

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

    def mean_pooling(self, model_output: torch.Tensor, attention_mask: torch.Tensor):
        # First element of model_output contains all token embeddings
        token_embeddings = model_output.encoder_last_hidden_state
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        )
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
            input_mask_expanded.sum(1), min=1e-9
        )

    def cls_pooling(self, model_output: torch.Tensor):
        return model_output.encoder_last_hidden_state[:, 0]

    def triplet_loss(self, target, pred, margin=1.0):
        pos_mask = target
        neg_mask = torch.abs(target - 1)
        n_pos = torch.sum(pos_mask, dim=-1)
        n_neg = torch.sum(neg_mask, dim=-1)
        sims_pos = pred * pos_mask
        sims_neg = pred * pos_mask
        mean_sim_pos = sims_pos / n_pos
        mean_sim_neg = sims_neg / n_neg
        return torch.mean(mean_sim_neg - mean_sim_pos + margin) * 10
