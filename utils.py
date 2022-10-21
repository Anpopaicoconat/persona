import os
import argparse
import json
import random

import torch
import pytorch_lightning as pl
import torchmetrics
import transformers


class PersonaDataset(torch.utils.data.Dataset):
    def __init__(self, path, mod, rnd_context=False, seed=42):
        super().__init__()
        self.data = []
        self.mod = mod
        self.rnd_context = rnd_context
        with open(path, "r") as file:
            for line in file:
                line = json.loads(line)
                if self.mod == "get_examples_gpt":
                    # generative for knowlede graunded generation
                    self.data += list(self.get_examples_gpt(**line))
                elif self.mod == "get_examples_candidat":
                    # simle ranker dialog agent
                    self.data += list(self.get_examples_candidat(**line))
                elif self.mod == "get_examples_gk":
                    # ranker for knowlede graunded generation
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
            persona = persons[dialog[i]["person"]]
            label = 1
            gks = [p for idx, p in enumerate(persona) if idx in dialog[i]["gk"]]
            for gk in gks:
                yield {
                    "context": context,
                    "gk": gk,
                    "label": label,
                }

    def get_examples_gpt(self, persons, dialog):
        dialog = self.join_same_person(dialog)
        for i in range(1, len(dialog)):
            if self.rnd_context:
                start = random.randint(0, i - 1)
            else:
                start = 0
            context = [t["text"] for t in dialog[start:i]]
            candidate = dialog[i]["text"]
            persona = persons[dialog[i]["person"]]
            gks = [p for idx, p in enumerate(persona) if idx in dialog[i]["gk"]]
            yield {
                "context": context,
                "candidate": candidate,
                "gk": gks,
            }

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class GenerativeCollator:
    def __init__(self, tokenizer, padding, max_length, return_tensors="pt"):
        self.tokenizer = tokenizer
        self.P1 = "[P1u]"  # user
        self.P2 = "[P2u]"  # model
        self.Gk = "[Gk]"
        self.eos = self.tokenizer.eos_token
        self.padding = padding
        self.max_length = max_length
        self.return_tensors = return_tensors

    def __call__(self, batch):
        for batch_id, example in enumerate(batch):
            # candidate
            output = self.eos
            output = self.P2 + example["candidate"] + output
            # gk
            for gk in example["gk"]:
                output = self.Gk + gk + output
            # context
            for i, context in enumerate(example["context"][::-1]):
                if i % 2 == 0:
                    P = self.P1
                else:
                    P = self.P2
                output = P + context + output
            batch[batch_id] = output

        batch = self.tokenizer.batch_encode_plus(
            batch,
            padding=self.padding,
            max_length=self.max_length,
            return_tensors=self.return_tensors,
            truncation=True,
        )
        return batch

    def test(self, batch):
        example = batch[0]
        # candidate
        input = self.eos
        input = self.P2 + example["candidate"] + input
        # gk
        for gk in example["gk"]:
            input = self.Gk + gk + input
        # context
        for i, context in enumerate(example["context"][::-1]):
            if i % 2 == 0:
                P = self.P1
            else:
                P = self.P2
            input = P + context + input

        input = self.tokenizer(
            input,
            padding=False,
            max_length=self.max_length,
            return_tensors=self.return_tensors,
            truncation=True,
        )
        sep_id = self.tokenizer(self.P2, return_tensors="pt")["input_ids"][0]
        mask = input["input_ids"][0] == sep_id
        try:
            mask = mask.nonzero()[-1][0]
        except:
            mask = -2

        return input, mask


class RetrievalCollator:
    def __init__(
        self,
        tokenizer,
        padding,
        max_length_context,
        max_length_candidate,
        return_tensors="pt",
    ):
        self.tokenizer = tokenizer
        self.P1 = "[P1u]"
        self.P2 = "[P2u]"
        self.Gk = "[Gk]"
        self.cls = tokenizer.cls_token
        self.eos = tokenizer.eos_token
        self.padding = padding
        self.max_length_context = max_length_context
        self.max_length_candidate = max_length_candidate
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
            max_length=self.max_length_context,
            return_tensors=self.return_tensors,
            truncation=True,
        )

    def CandidateCollator(self, batch):
        return self.tokenizer.batch_encode_plus(
            batch,
            padding=self.padding,
            max_length=self.max_length_candidate,
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
            max_length=self.max_length_candidate,
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
        out = torch.mm(x, y.transpose(0, 1)) * 10
    return out
