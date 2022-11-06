import json
import random

import torch
import pytorch_lightning as pl


class Toloka_DS(torch.utils.data.Dataset):
    def __init__(self, path, exaples, context_len=False, seed=42):
        """_summary_

        Args:
            path (_type_): _description_
            exaples (str): answer - контекст+ответ,
                           all_gk - контекст+gk 1...контекст+gk n,
                           one_gk - контекст+ случайный gk n.
                           за 1 эпоху
            context_len (bool, optional): rnd - случайная длинна контекста до кандидата
                                          all - весь доступный контекст
                                          one - 1 часть доступного контекста
            seed (int, optional): _description_. Defaults to 42.
        """
        super().__init__()
        self.data = []
        self.exaples = exaples
        self.context_len = context_len
        with open(path, "r") as file:
            for line in file:
                line = json.loads(line)
                self.data += list(self.get_examples(**line))

    def join_same_person(self, dialog):
        new_dialog = dialog[:1]
        for d in dialog[1:]:
            if new_dialog[-1]["person"] == d["person"]:
                new_dialog[-1]["text"] = new_dialog[-1]["text"] + " " + d["text"]
                new_dialog[-1]["gk"] = list(set(new_dialog[-1]["gk"]) | set(d["gk"]))
            else:
                new_dialog.append(d)
        return new_dialog

    def get_examples(self, persons, dialog):
        dialog = self.join_same_person(dialog)
        for i in range(1, len(dialog)):
            if self.ex_per_dialog == 'rnd':
                i = random.randint(1, len(dialog))

            if self.context_len == "rnd":
                start = random.randint(0, i - 1)
            elif self.context_len == "all":
                start = 0
            elif self.context_len == "one":
                start = i - 1
            context = [t["text"] for t in dialog[start:i]]
            candidate = dialog[i]["text"]
            persona = persons[dialog[i]["person"]]
            label = 1
            if self.exaples == "answer":
                yield {
                    "context": context,
                    "answer": candidate,
                    "persona": persona,
                    "label": label,
                }
            elif self.exaples[-2:] == "gk":
                # gks = [p for idx, p in enumerate(persona) if idx in dialog[i]["gk"]]
                gks = [persona[idx] for idx in dialog[i]["gk"]]
                if self.exaples == "one_gk" and gks:
                    # что бы избежать попадания gk одного контекста в негативные берем 1 gk в эпоху
                    gks = [random.choice(gks)]
                for gk in gks:
                    yield {
                        "context": context,
                        "gk": gk,
                        "label": label,
                    }
            if self.ex_per_dialog == 'rnd':
                break

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


spec_tokens = [
    "[Model]",
    "[User]",
    "[MaleG]",
    "[FemaleG]",
    "[UnknownG]",
    "[ModelGK]",
    "[UserGK]",
    "[WorldGK]",
    "|DialogContext|:",
    "|DialogAnswer|:",
    "|DialogModelGK|:",
    "|DialogCrossEnc|:",
]


class Collator:
    def __init__(
        self,
        spectokens,
        tokenizer,
        padding,
        qury_len,
        cand_len,
        return_tensors="pt",
    ):
        self.tokenizer = tokenizer
        (
            self.model_token,
            self.user_token,
            self.m_gender,
            self.f_gender,
            self.u_gender,
            self.model_gk,
            self.user_gk,
            self.world_gk,
            self.DialogContext,
            self.DialogAnswer,
            self.DialogModelGK,
            self.DialogCrossEnc,
        ) = spectokens
        self.padding = padding
        self.qury_len = qury_len
        self.cand_len = cand_len
        self.return_tensors = return_tensors

    def rebatch(self, batch):
        batch_new = {k: [] for k in batch[0]}
        for example in batch:
            for k in example:
                batch_new[k].append(example[k])
        return batch_new

    def BiEncoder(self, batch):
        batch = self.rebatch(batch)
        for k in batch:
            if k == "context":
                batch[k] = self.Context(batch[k], self.DialogContext)
            elif k == "answer":
                batch[k] = self.Candidat(batch[k], self.DialogAnswer)
            elif k == "gk":
                batch[k] = self.Candidat(batch[k], self.DialogModelGK)

        return batch

    def Context(self, batch, prefix):
        for b_i, context in enumerate(batch):
            c_out = self.model_token
            for i, c in enumerate(context[::-1]):
                if i % 2 == 0:
                    P = self.user_token
                else:
                    P = self.model_token
                c_out = P + c + c_out
            batch[b_i] = c_out
        prefix = self.tokenizer(
            [prefix], add_special_tokens=False, return_tensors=self.return_tensors
        )
        encoded_batch = self.tokenizer(
            batch,
            padding=self.padding,
            max_length=self.qury_len - len(prefix["input_ids"]),
            return_tensors=self.return_tensors,
            truncation=True,
        )
        # заменяем первые токены префиксом что бы макс динна не менялась
        for k in encoded_batch:
            encoded_batch[k] = torch.concat(
                [prefix[k].repeat((encoded_batch[k].size()[0], 1)), encoded_batch[k]],
                dim=1,
            )
        return encoded_batch

    def Candidat(self, batch, prefix):
        prefix = self.tokenizer(
            [prefix], add_special_tokens=False, return_tensors=self.return_tensors
        )
        encoded_batch = self.tokenizer(
            batch,
            padding=self.padding,
            max_length=self.cand_len - len(prefix["input_ids"]),
            return_tensors=self.return_tensors,
            truncation=True,
        )
        for k in encoded_batch:
            encoded_batch[k] = torch.concat(
                [prefix[k].repeat((encoded_batch[k].size()[0], 1)), encoded_batch[k]],
                dim=1,
            )
        return encoded_batch
