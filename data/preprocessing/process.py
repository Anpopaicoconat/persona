import json
import re
import numpy as np
import pandas as pd


def proc_dialog(inp):
    inp = re.sub("<\/span>|Пользователь \w:", " ", inp)
    inp = re.sub("<br \/>", " ", inp)  # \n
    inp = re.split("<span class=participant_", inp)[1:]
    out = []
    for turn in inp:
        current_speaker, turn = int(turn[0]), turn[2:]
        turn = re.sub("\ {2,}", " ", turn).strip()
        out.append({"person": int(current_speaker) - 1, "text": turn})
    return out


def proc_persona(inp):
    inp = re.sub("<span class=participant_\w>|<\/span>", "", inp)
    inp = re.split("<br \/>", inp)[:-1]
    return inp


def proc_row(row):
    dialog = proc_dialog(row.dialogue)
    persons = [proc_persona(p) for p in [row.persona_1_profile, row.persona_2_profile]]
    return persons, dialog


def proc(raw_path, out_path):
    df = pd.read_csv(raw_path, sep="\t")

    with open(out_path, "w") as out_file:
        for row in df.iterrows():
            row = row[1]
            persons, dialog = proc_row(row)
            out = {"persons": persons, "dialog": dialog}
            out_file.write(json.dumps(out, ensure_ascii=False) + "\n")


proc("TlkPersonaChatRus/dialogues.tsv", "TlkPersonaChatRus/TolokaPersonaChat.jsonl")
