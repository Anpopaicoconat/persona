import json
from datetime import datetime
import re
import pprint
import random
import numpy as np
import pandas as pd

import torch
import pytorch_lightning as pl
import torchmetrics
import transformers

import os

os.environ["http_proxy"] = "http://proxy.ad.speechpro.com:3128"
os.environ["https_proxy"] = "http://proxy.ad.speechpro.com:3128"
os.environ["ftp_proxy"] = "http://proxy.ad.speechpro.com:3128"


class PersonaRetrievalDataset(torch.utils.data.Dataset):
    def __init__(self, path, rnd_context=False, seed=42):
        super().__init__()
        self.data = []
        self.rnd_context = rnd_context
        with open(path, "r") as file:
            for line in file:
                line = json.loads(line)
                self.data += list(self.get_examples(**line))

    def join_same_person(dialog):
        new_dialog = dialog[:1]
        for d in dialog[1:]:
            if new_dialog[-1]["person"] == d["person"]:
                new_dialog[-1]["text"] = new_dialog[-1]["text"] + " " + d["person"]
            else:
                new_dialog.append(d)

    def get_examples(self, persons, dialog):
        for i in range(1, len(dialog)):
            if self.rnd_context:
                start = random.randint(0, i - 1)
            else:
                start = 0
            context = [t["text"] for t in dialog[start:i]]
            candidate = dialog[i]["text"]
            persona = persons[dialog[i]["person"]]
            label = 1

            yield {
                "context": context,
                "candidate": candidate,
                "persona": persona,
                "label": label,
            }

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class RetrievalCollator:
    def __init__(self, tokenizer, padding, max_length, return_tensors="pt"):
        self.tokenizer = tokenizer
        self.P1 = "[P1u]"
        self.P2 = "[P2u]"
        self.Gk = "[Gk]"
        self.cls = tokenizer.cls_token
        self.eos = tokenizer.eos_token
        self.padding = padding
        self.max_length = max_length
        self.return_tensors = return_tensors

    def __call__(self, batch):
        batch_new = {k: [] for k in batch[0]}
        for example in batch:
            for k in example:
                batch_new[k].append(example[k])
        batch_new["context"] = self.ContextCollator(batch_new["context"])
        batch_new["candidate"] = self.CandidateCollator(batch_new["candidate"])
        batch_new["persona"] = self.PersonaCollator(batch_new["persona"])
        return batch_new

    def ContextCollator(self, batch):
        for i, context in enumerate(batch):
            c_out = self.P2
            for c in context[::-1]:
                if i % 2 == 0:
                    P = self.P1
                else:
                    P = self.P2
                c_out = P + c + c_out
            batch[i] = c_out
        return self.tokenizer.batch_encode_plus(
            batch,
            padding=self.padding,
            max_length=self.max_length,
            return_tensors=self.return_tensors,
            truncation=True,
        )

    def CandidateCollator(self, batch):
        return self.tokenizer.batch_encode_plus(
            batch,
            padding=self.padding,
            max_length=self.max_length,
            return_tensors=self.return_tensors,
            truncation=True,
        )

    def PersonaCollator(self, batch):
        for i, persona in enumerate(batch):
            c_out = self.Gk
            for c in persona[::-1]:
                c_out = self.Gk + c + c_out
            batch[i] = c_out
        return self.tokenizer.batch_encode_plus(
            batch,
            padding=self.padding,
            max_length=self.max_length,
            return_tensors=self.return_tensors,
            truncation=True,
        )


def aggregate_encoder_output(
    model_output,
    mod: str,
):
    if mod == "pooler_output":
        out = model_output.pooler_output
    elif mod == "last_hidden_state_cls_left":
        out = model_output.last_hidden_state[:, 0, :]
    elif mod == "last_hidden_state_cls_right":
        out = model_output.last_hidden_state[:, -1, :]
    elif mod == "last_hidden_state_mean":
        # TODO проверить нужно ли маскирование
        out = torch.mean(out.last_hidden_state, dim=1)
    return out


def sim_func(x, y, mod):
    if mod == "DotProduct":
        out = torch.mm(x, y.transpose(0, 1))
    elif mod == "CosineSimilarity":
        x = x / x.norm(dim=1)[:, None]
        y = y / y.norm(dim=1)[:, None]
        out = torch.mm(x, y.transpose(0, 1))
    return out


class RetrievalModel(pl.LightningModule):
    def __init__(
        self,
        context_BERT,
        candidat_BERT,
        batch_size,
        scheduler_len,
        num_warmup_steps,
        lr,
    ):
        super().__init__()
        self.context_BERT = context_BERT
        self.candidat_BERT = candidat_BERT
        self.batch_size = batch_size
        self.scheduler_len = scheduler_len
        self.num_warmup_steps = num_warmup_steps
        self.loss = torch.nn.CrossEntropyLoss()
        self.lr = lr
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

    def training_step(self, batch, batch_idx):
        context = batch["context"]
        candidate = batch["candidate"]
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

        return loss

    def training_epoch_end(self, outputs):
        self.train_metrics.reset()

    def validation_step(self, val_batch, batch_idx):
        context = val_batch["context"]
        candidate = val_batch["candidate"]
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

        return val_metrics, val_loss

    def training_epoch_end(self, outputs):
        self.val_metrics.reset()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        scheduler = transformers.get_cosine_schedule_with_warmup(
            optimizer, num_warmup_steps=5000, num_training_steps=self.scheduler_len
        )
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]

    def forward(self, context, candidat, labels):
        context_vec = self.context_BERT(**context)
        candidat_vec = self.candidat_BERT(**candidat)
        context_vec = aggregate_encoder_output(
            context_vec, mod="last_hidden_state_cls_left"
        )
        candidat_vec = aggregate_encoder_output(
            candidat_vec, mod="last_hidden_state_cls_left"
        )
        distance = sim_func(context_vec, candidat_vec, "DotProduct")
        return distance


epochs = 15
lr = 7e-5
batch_size = 64
context_len = 128
candidate_len = 64
persona_len = 2
val_split = 5

pretrained_path = "/home/stc/persona/models/rubert-base-cased-conversational"
data_path = "/home/stc/persona/data/TlkPersonaChatRus/TolokaPersonaChat.jsonl"

tokenizer = transformers.AutoTokenizer.from_pretrained(
    pretrained_path, truncation_side="left", padding_side="right"
)
special_tokens_dict = {
    "additional_special_tokens": [
        "[P1x]",
        "[P1x]",
        "[P2y]",
        "[P2y]",
        "[P1u]",
        "[P2u]",
        "[Gk]",
    ]
}
tokenizer.add_special_tokens(special_tokens_dict)
# [P1x] P-turn start, 1-user, 2-model, x-male, y-female, u-unknown
context_bert = transformers.AutoModel.from_pretrained(pretrained_path)
context_bert.resize_token_embeddings(len(tokenizer))
candidate_bert = transformers.AutoModel.from_pretrained(pretrained_path)
candidate_bert.resize_token_embeddings(len(tokenizer))

dataset = PersonaRetrievalDataset(data_path)
train_dataset, val_dataset = torch.utils.data.random_split(
    dataset, [len(dataset) - (len(dataset) // val_split), (len(dataset) // val_split)]
)

callator = RetrievalCollator(tokenizer, padding="max_length", max_length=context_len)

train_dataloader = torch.utils.data.DataLoader(
    train_dataset, batch_size=batch_size, shuffle=False, collate_fn=callator
)
val_dataloader = torch.utils.data.DataLoader(
    val_dataset, batch_size=batch_size, shuffle=False, collate_fn=callator
)

scheduler_len = len(train_dataloader) * epochs
num_warmup_steps = 5000

model = RetrievalModel(
    context_bert, candidate_bert, batch_size, scheduler_len, num_warmup_steps, lr
)
logger = pl.loggers.comet.CometLogger(
    api_key="sEJsZrYjwc0gxxUAUGQNBwTsb",
    save_dir="logs",
    project_name="bi_encoder",
    experiment_name="nopersona_answers",
)
trainer = pl.Trainer(
    max_epochs=epochs,
    accelerator="gpu",
    devices=1,
    gradient_clip_val=1,
    logger=logger,
    num_sanity_val_steps=0,
)
trainer.fit(model, train_dataloader, val_dataloader)
