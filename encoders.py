import os
import argparse
import json
import random

import torch
import pytorch_lightning as pl
import torchmetrics
import transformers

parser = argparse.ArgumentParser()
parser.add_argument("-epochs", help="num of epochs for train", type=int, default=10)
parser.add_argument("-lr", help="learning rate", type=float, default=1e-5)
parser.add_argument("-batch_size", help="batch size", type=int, default=64)
parser.add_argument("-context_len", help="context len", type=int, default=64)
parser.add_argument("-candidate_len", help="candidate len", type=int, default=64)
parser.add_argument("-persona_len", help="persona len", type=int, default=64)
parser.add_argument("-val_split", help="val split", type=int, default=64)
parser.add_argument(
    "-truncation_side", help="truncation side", type=str, default="left"
)
parser.add_argument("-padding_side", help="padding side", type=str, default="right")
parser.add_argument("-gradient_clip_val", help="gradient clip val", type=int, default=1)
parser.add_argument(
    "-num_warmup_steps", help="num warmup steps", type=int, default=1000
)
parser.add_argument(
    "-pretrained_path",
    help="pretrained path",
    type=str,
    default="/home/posokhov@ad.speechpro.com/projects/models/conversational/",
)
parser.add_argument(
    "-data_path",
    help="data path",
    type=str,
    default="data/TlkPersonaChatRus/TolokaPersonaChat.jsonl",
)
parser.add_argument(
    "-project_name",
    help="project name",
    type=str,
    default="bi-encoder",
)
parser.add_argument(
    "-experiment_name",
    help="experiment name",
    type=str,
    default="test",
)
args = parser.parse_args()
with open("config.json", "r") as read_content:
    opt = json.load(read_content)
vars(args).update(opt)

epochs = args.epochs
lr = args.lr
batch_size = args.batch_size
context_len = args.context_len
candidate_len = args.candidate_len
persona_len = args.persona_len
val_split = args.val_split
truncation_side = args.truncation_side
padding_side = args.padding_side
gradient_clip_val = args.gradient_clip_val
num_warmup_steps = args.num_warmup_steps
pretrained_path = args.pretrained_path
data_path = args.data_path
project_name = args.project_name
experiment_name = args.experiment_name

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
                self.data += list(self.get_examples_gk(**line))

    def join_same_person(self, dialog):
        new_dialog = dialog[:1]
        for d in dialog[1:]:
            if new_dialog[-1]["person"] == d["person"]:
                new_dialog[-1]["text"] = new_dialog[-1]["text"] + " " + d["text"]
                new_dialog[-1]["gk"] = list(set(new_dialog[-1]["gk"]) | set(d["gk"]))
            else:
                new_dialog.append(d)
        return new_dialog

    def get_examples_candidat(self, persons, dialog):
        dialog = self.join_same_person(dialog)
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

    def get_examples_gk(self, persons, dialog):
        dialog = self.join_same_person(dialog)
        for i in range(1, len(dialog)):
            if self.rnd_context:
                start = random.randint(0, i - 1)
            else:
                start = 0
            context = [t["text"] for t in dialog[start:i]]
            candidate = dialog[i]["text"]
            persona = persons[dialog[i]["person"]]
            label = 1
            gks = [p for idx, p in enumerate(persona) if idx in dialog[i]["gk"]]
            for gk in gks:
                yield {
                    "context": context,
                    # "candidate": candidate,
                    "gk": gk,
                    # "persona": persona,
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
        # batch_new["candidate"] = self.CandidateCollator(batch_new["candidate"])
        batch_new["gk"] = self.CandidateCollator(batch_new["gk"])
        # batch_new["persona"] = self.PersonaCollator(batch_new["persona"])
        return batch_new

    def ContextCollator(self, batch):
        for b_i, context in enumerate(batch):
            c_out = self.P2
            for i, c in enumerate(context[::-1]):
                if i % 2 == 0:
                    P = self.P1
                else:
                    P = self.P2
                c_out = P + c + c_out
            batch[b_i] = c_out
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

        return loss

    def training_epoch_end(self, outputs):
        self.train_metrics.reset()

    def validation_step(self, val_batch, batch_idx):
        context = val_batch["context"]
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
            context_vec, mod="last_hidden_state_cls_left"  # add arg
        )
        candidat_vec = aggregate_encoder_output(
            candidat_vec, mod="last_hidden_state_cls_left"
        )
        distance = sim_func(context_vec, candidat_vec, "DotProduct")
        return distance


#########################################################

# tokenizer = transformers.AutoTokenizer.from_pretrained(
#     pretrained_path, truncation_side=truncation_side, padding_side=padding_side
# )
# special_tokens_dict = {
#     "additional_special_tokens": [
#         "[P1m]",
#         "[P1m]",
#         "[P2f]",
#         "[P2f]",
#         "[P1u]",
#         "[P2u]",
#         "[Gk]",
#     ]
# }
# tokenizer.add_special_tokens(special_tokens_dict)
# # [P1x] P-turn start, 1-user, 2-model, m-male, f-female, u-unknown
# context_bert = transformers.AutoModel.from_pretrained(pretrained_path)
# context_bert.resize_token_embeddings(len(tokenizer))
# candidate_bert = transformers.AutoModel.from_pretrained(pretrained_path)
# candidate_bert.resize_token_embeddings(len(tokenizer))

# dataset = PersonaRetrievalDataset(data_path)
# train_dataset, val_dataset = torch.utils.data.random_split(
#     dataset, [len(dataset) - (len(dataset) // val_split), (len(dataset) // val_split)]
# )

# callator = RetrievalCollator(tokenizer, padding="max_length", max_length=context_len)

# train_dataloader = torch.utils.data.DataLoader(
#     train_dataset, batch_size=batch_size, shuffle=True, collate_fn=callator
# )
# val_dataloader = torch.utils.data.DataLoader(
#     val_dataset, batch_size=batch_size, shuffle=False, collate_fn=callator
# )

# scheduler_len = len(train_dataloader) * epochs

# model = RetrievalModel(
#     context_bert, candidate_bert, batch_size, scheduler_len, num_warmup_steps, lr
# )
# logger = pl.loggers.comet.CometLogger(
#     api_key="sEJsZrYjwc0gxxUAUGQNBwTsb",
#     save_dir="logs",
#     project_name=project_name,
#     experiment_name=experiment_name,
# )
# logger.log_hyperparams(args)
# trainer = pl.Trainer(
#     max_epochs=epochs,
#     accelerator="gpu",
#     devices=1,
#     gradient_clip_val=gradient_clip_val,
#     logger=logger,
#     num_sanity_val_steps=1,
# )
# trainer.fit(model, train_dataloader, val_dataloader)
