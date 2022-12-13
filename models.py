from typing import *

import torch
import pytorch_lightning as pl
import torchmetrics
import transformers


class single_multitask_model(pl.LightningModule):
    """_summary_

    Attributes:
        transformer (): model
        tokenizers (): tokenizer
        scheduler_len (int): len dataloder * num epochs
        num_warmup_steps (int): len of warmup
        lr (int): learning rate
        batch_size (int): size of batch ?единый для всех тасков?
    """

    def __init__(
        self,
        transformer,
        tokenizer,
        collator,
        pooling: str,
        scheduler_len: int,
        num_warmup_steps: int,
        lr: float,
    ):
        super().__init__()
        self.transformer = transformer
        self.tokenizers = tokenizer
        self.collator = collator

        self.pooling = pooling
        self.scale = 20

        self.scheduler_len = scheduler_len
        self.num_warmup_steps = num_warmup_steps
        self.lr = lr
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss()

        # BiEncoder
        self.bienc_answer_loss = torch.nn.CrossEntropyLoss()
        self.bienc_gk_loss = torch.nn.CrossEntropyLoss()
        bienc_metrics = torchmetrics.MetricCollection(
            {
                "r1": torchmetrics.RetrievalRecall(k=1),
                "r5": torchmetrics.RetrievalRecall(k=5),
                "mrr": torchmetrics.RetrievalMRR(),
            }
        )
        self.train_bienc_answer_metrics = bienc_metrics.clone(prefix="train_answer_")
        self.train_bienc_gk_metrics = bienc_metrics.clone(prefix="train_gk_")
        self.val_bienc_answer_metrics = bienc_metrics.clone(prefix="val_answer_")
        self.val_bienc_gk_metrics = bienc_metrics.clone(prefix="val_gk_")

        # Generative
        self.gen_metrics = torchmetrics.MetricCollection(
            {
                "val_BLEU1": torchmetrics.BLEUScore(n_gram=1),
                "val_BLEU2": torchmetrics.BLEUScore(n_gram=2),
                "val_BLEU4": torchmetrics.BLEUScore(n_gram=4),
            }
        )

        self.save_hyperparameters()

    def encode(
        self,
        batch: Dict,
    ):
        model_output = self.t5(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["input_ids"],
            output_hidden_states=True,
        )
        model_output = model_output.encoder_last_hidden_state

        if self.pooling == "mean":
            embeddings = self.mean_pooling(model_output, batch["attention_mask"])
        elif self.pooling == "cls":
            embeddings = self.cls_pooling(model_output)

        if self.args.distance == "cosine":
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

        return embeddings

    def generate(
        self,
        input_batch: Dict,
        output_batch: Dict,
    ):
        model_output = self.t5(
            input_ids=input_batch["input_ids"],
            attention_mask=input_batch["attention_mask"],
            labels=output_batch["input_ids"],
        )

        return model_output

    def training_step(self, batch, batch_idx):
        if batch["type"] == "rank_next_gk":
            q_embedding = self.encode(batch["batch"]["query"])
            c_embedding = self.encode(batch["batch"]["candidate"])
            scores = torch.mm(q_embedding, c_embedding.transpose(0, 1)) * self.scale
            labels = torch.tensor(
                range(len(scores)), dtype=torch.long, device=q_embedding.device
            )
            loss = (
                self.cross_entropy_loss(scores, labels)
                + self.cross_entropy_loss(scores.transpose(0, 1), labels)
            ) / 2

        elif batch["type"] == "rank_answer":
            q_embedding = self.encode(batch["batch"]["query"])
            c_embedding = self.encode(batch["batch"]["candidate"])
            scores = torch.mm(q_embedding, c_embedding.transpose(0, 1)) * self.scale
            labels = torch.tensor(
                range(len(scores)), dtype=torch.long, device=q_embedding.device
            )
            loss = (
                self.cross_entropy_loss(scores, labels)
                + self.cross_entropy_loss(scores.transpose(0, 1), labels)
            ) / 2

        elif batch["type"] == "generate_answer":
            model_output = self.generate(
                batch["batch"]["query"], batch["batch"]["candidate"]
            )
            loss = model_output.loss

        elif batch["type"] == "generate_current_gk":
            model_output = self.generate(
                batch["batch"]["query"], batch["batch"]["candidate"]
            )
            loss = model_output.loss

        return loss

    def validation_step(self, batch, batch_idx):
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        scheduler = transformers.get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.num_warmup_steps,
            num_training_steps=self.scheduler_len,
        )
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]

    def mean_pooling(model_output: torch.Tensor, attention_mask: torch.Tensor):
        token_embeddings = model_output
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        )
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
            input_mask_expanded.sum(1), min=1e-9
        )

    def cls_pooling(self, model_output: torch.Tensor):
        return model_output
