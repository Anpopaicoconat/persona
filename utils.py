from typing import *
import json
import random

import torch


class TolokaDataset(torch.utils.data.Dataset):
    def __init__(self, path, task, query_len="all", seed=42):
        """_summary_

        Args:
            path (_type_): _description_
            task (_type_): rank_next_gk, rank_answer, generate_answer, generate_current_gk
            query_len (str, optional): _description_. Defaults to "all".
            seed (int, optional): _description_. Defaults to 42.
        """
        with open(path, "r") as file:
            self.dialogs = []
            for line in file:
                line = json.loads(line)
                if task != "generate_current_gk":
                    line = self.join_same_person(**line)
                self.dialogs.append(line)
        self.task = task
        self.query_len = query_len

    def __len__(self) -> int:
        return len(self.dialogs)

    def __getitem__(self, idx):
        dialog = self.dialogs[idx]["dialog"]
        persons = self.dialogs[idx]["persons"]

        if self.task == "rank_next_gk":
            valid_idx = [i for i, turn in enumerate(dialog) if turn["gk"] and i]
        elif self.task == "generate_current_gk":
            valid_idx = [i for i, turn in enumerate(dialog) if turn["gk"]]
        elif self.task == "rank_answer" or "generate_answer":
            valid_idx = [i for i, turn in enumerate(dialog) if i]
        # print(valid_idx)
        end = random.choice(valid_idx)

        if self.query_len == "rnd":
            start = random.randint(0, end - 1)
        elif self.query_len == "all":
            start = 0
        elif self.query_len == "one":
            start = end - 1
        persona = persons[dialog[end]["person"]]

        if self.task == "rank_next_gk":
            query = [
                {"gender": persons[t["person"]]["gender"], "text": t["text"]}
                for t in dialog[start:end]
            ]
            candidate = {
                "text": random.choice(
                    [persona["description"][i] for i in dialog[end]["gk"]]
                )
            }
        elif self.task == "generate_current_gk":
            query = [{"gender": persona["gender"], "text": dialog[end]["text"]}]
            candidate = {
                "text": random.choice(
                    [persona["description"][i] for i in dialog[end]["gk"]]
                )
            }
        elif self.task == "rank_answer" or "generate_answer":
            query = [
                {"gender": persons[t["person"]]["gender"], "text": t["text"]}
                for t in dialog[start:end]
            ]
            gk = [persona["description"][i] for i in dialog[end]["gk"]]
            candidate = {"gender": persona["gender"], "text": dialog[end]["text"]}
            return {"query": query, "gk": gk, "candidate": candidate}

        return {"query": query, "candidate": candidate}

    def join_same_person(self, persons: Dict, dialog: Dict):
        new_dialog = dialog[:1]
        for d in dialog[1:]:
            if new_dialog[-1]["person"] == d["person"]:
                new_dialog[-1]["text"] = new_dialog[-1]["text"] + " " + d["text"]
                new_dialog[-1]["gk"] = list(set(new_dialog[-1]["gk"]) | set(d["gk"]))
            else:
                new_dialog.append(d)
        return {"persons": persons, "dialog": new_dialog}
