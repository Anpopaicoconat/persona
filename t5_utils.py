import json
import random

import torch
import pytorch_lightning as pl


class Toloka_DS(torch.utils.data.Dataset):
    def __init__(self, path, exaples, ex_per_dialog, context_len=False, seed=42):
        """_summary_

        Args:
            path (_type_): _description_
            exaples (str): answer - контекст+ответ,
                           all_gk - контекст+gk 1...контекст+gk n,
                           one_gk - контекст+ случайный gk n.
                           за 1 эпоху
            ex_per_dialog (str): all - все реплики из диалога в эпохе,
                           rnd - одна реплика из диалога в эпохе.
            context_len (bool, optional): rnd - случайная длинна контекста до кандидата
                                          all - весь доступный контекст
                                          one - 1 часть доступного контекста
            seed (int, optional): _description_. Defaults to 42.
        """
        super().__init__()
        self.data = []
        self.exaples = exaples
        self.context_len = context_len
        self.ex_per_dialog = ex_per_dialog
        self.prefix_context = "|DialogContext|:"
        self.prefix_answer = "|DialogAnswer|:"
        self.prefix_gk = "|DialogModelGK|:"
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
            if self.ex_per_dialog == "rnd":
                i = random.randint(1, len(dialog) - 1)

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
            gks = [persona[idx] for idx in dialog[i]["gk"]]
            if self.exaples == "answer":
                yield {
                    "query": {"content": context, "prefix": self.prefix_context},
                    "candidat": {"content": candidate, "prefix": self.prefix_answer},
                    "gks": {"content": gks, "prefix": self.prefix_gk},
                }
            elif self.exaples[-2:] == "gk":
                # gks = [p for idx, p in enumerate(persona) if idx in dialog[i]["gk"]]
                if self.exaples == "one_gk" and gks:
                    # что бы избежать попадания gk одного контекста в негативные берем 1 gk в эпоху
                    gks = [random.choice(gks)]
                for gk in gks:
                    yield {
                        "query": {"content": context, "prefix": self.prefix_context},
                        "candidat": {"content": gk, "prefix": self.prefix_gk},
                    }
            if self.ex_per_dialog == "rnd":
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
                batch_new[k].append(example[k]["content"])
        return batch_new

    def BiEncoder(self, batch):
        context_prefix = batch[0]["query"]["prefix"]
        candidat_prefix = batch[0]["candidat"]["prefix"]
        batch = self.rebatch(batch)
        batch["query"] = self.Context(batch["query"], context_prefix)
        batch["candidat"] = self.Candidat(batch["candidat"], candidat_prefix)
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


class SequntiaLoader:
    """This class combines dataloaders and returns single batch from one of them per iteration
    Attributes:
        dataloders_dict (dict): dict with next structure:
                key (str) - batch type that will be returned,
                value (torch.utils.data.DataLoader) - dataloader.
        lens_dict (dict): dict with next structure:
                key (str) - batch type,
                value (int) - length of single dataloader - k
        full_len (int): full length of all dataloaders
        batch_ogorder (list) - list of batch keys in sequential order
        shuffle (bool) boolean that states elements order in batch:
                True - returns batch from randomly chosen dataloader each iteration
                False - returns batch from each dataloader sequentialy
        batch_order (list) - current order of batch keys either shuffled or sequential which generates on each itteration loop
        dataloders (dict) - current dict of iterable dataloders generates for each itteration loop
    """

    def __init__(self, dataloders: dict, shuffle: bool):
        """
        init:
            dict containing: lengths of all dataloaders; full length of all dataloaders; sequential batch order
        Args:
            dataloders (dict): dict with following structure:
                key - batch type that will be returned with batch content,
                value - torch.utils.data.DataLoader
            shuffle (bool) boolean that states elements order in batch:
                True - returns batch from randomly chosen dataloader per iteration,
                False - returns batch from each dataloader sequentialy
        """
        self.dataloders_dict = dataloders
        self.lens_dict = {k: len(self.dataloders_dict[k]) for k in self.dataloders_dict}
        self.full_len = sum([self.lens_dict[k] for k in self.lens_dict])
        self.batch_ogorder = []
        for k in self.lens_dict:
            self.batch_ogorder += [k for _ in range(self.lens_dict[k])]
        self.shuffle = shuffle

    def __iter__(self) -> iter:
        """init batch order and make dataloaders iterable

        Returns:
            iter: itterable object with chosen batch order
        """
        self.batch_order = self.batch_ogorder
        if self.shuffle:
            random.shuffle(self.batch_order)

        self.dataloders = {
            k: iter(self.dataloders_dict[k]) for k in self.dataloders_dict
        }
        self.batch_order = iter(self.batch_order)
        return self

    def __len__(self) -> int:
        """returns full length of all dataloaders

        Returns:
            int: full length of all dataloaders
        """
        return self.full_len

    def __next__(self) -> dict:
        """returns batch from one of dataloders in chosen order

        Raises:
            StopIteration: when all dataloaders was iterated

        Returns:
            dict: with structure:
                batch (any) - batch from dataloder
                type (str) - key of dataloder from which batch is taken from
        """
        try:
            k = next(self.batch_order)
            return {"batch": next(self.dataloders[k]), "type": k}
        except IndexError:
            raise StopIteration
