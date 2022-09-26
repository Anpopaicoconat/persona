import os
import argparse
import json

import torch
import pytorch_lightning as pl
import torchmetrics
import transformers

from utils import (
    PersonaDataset,
    GenerativeCollator,
    RetrievalCollator,
    aggregate_encoder_output,
    sim_func,
)
from models import RetrievalModel, GenerativeModel

os.environ["http_proxy"] = "http://proxy.ad.speechpro.com:3128"
os.environ["https_proxy"] = "http://proxy.ad.speechpro.com:3128"
os.environ["ftp_proxy"] = "http://proxy.ad.speechpro.com:3128"

# config bert
parser = argparse.ArgumentParser()
bert_args = parser.parse_args("")
with open("configs/bert_config.json", "r") as config:
    opt = json.load(config)
vars(bert_args).update(opt)
# config gpt
parser = argparse.ArgumentParser()
gpt_args = parser.parse_args("")
with open("configs/gpt_config.json", "r") as config:
    opt = json.load(config)
vars(gpt_args).update(opt)

# tokenizer bert
with open(bert_args.special_tokens_dict, "r") as config:
    special_tokens_dict = json.load(config)

bert_tokenizer = transformers.AutoTokenizer.from_pretrained(
    bert_args.pretrained_bert,
    truncation_side=bert_args.truncation_side,
    padding_side=bert_args.padding_side,
)
bert_tokenizer.add_special_tokens(special_tokens_dict)
# tokenizer gpt
with open(gpt_args.special_tokens_dict, "r") as config:
    special_tokens_dict = json.load(config)

gpt_tokenizer = transformers.AutoTokenizer.from_pretrained(
    gpt_args.pretrained_gpt,
    truncation_side=gpt_args.truncation_side,
    padding_side=gpt_args.padding_side,
)
gpt_tokenizer.pad_token = gpt_tokenizer.eos_token
gpt_tokenizer.add_special_tokens(special_tokens_dict)

# callator bert
bert_callator = RetrievalCollator(
    bert_tokenizer, padding=bert_args.padding, max_length=bert_args.context_len
)
# callator gpt
gpt_callator = GenerativeCollator(
    gpt_tokenizer, padding=gpt_args.padding, max_length=gpt_args.max_len
)


# retrieva model
bert_ckpt = "/home/stc/persona/logs/bi_encoder/best/checkpoints/epoch=5-step=4554.ckpt"
retrieval_model = RetrievalModel.load_from_checkpoint(bert_ckpt)
# generative model
gpt_ckpt = "/home/stc/persona/logs/gpt_answer/1ep.ckpt"
generative_model = GenerativeModel.load_from_checkpoint(gpt_ckpt)

# encode functions
def encode_persona(text_batch, encoder):
    inp_persona_tokens = bert_callator.CandidateCollator(text_batch)
    vec_batch = aggregate_encoder_output(
        encoder.candidat_BERT(**inp_persona_tokens), mod=bert_args.aggregation_mod
    )
    return vec_batch


def encode_context(text_batch, encoder):
    inp_context_tokens = bert_callator.ContextCollator([text_batch])
    print(bert_tokenizer.batch_decode(inp_context_tokens["input_ids"]))
    vec_batch = aggregate_encoder_output(
        encoder.context_BERT(**inp_context_tokens), mod=bert_args.aggregation_mod
    )
    return vec_batch


persona = [
    "У меня любимая работа.",
    "Я уважаю людей.",
    "У меня есть попугай.",
    "Я был в Париже.",
    "Я люблю кофе.",
    "У меня есть собака.",
    "У меня есть кошка.",
    "Я играю на гитаре.",
    "Я кассир.",
    "Я работаю в магазине.",
]
context = ["привет"]

vec_persona = encode_persona(persona, retrieval_model)
user_msg = ""
for c in context:
    print(c)
while True:
    user_msg = input("user: ")
    if user_msg == "!stop":
        break
    if user_msg == "!clear":
        context = []
        continue

    context.append(user_msg)
    vec_context = encode_context(context, retrieval_model)

    ranks = sim_func(vec_context, vec_persona, mod="DotProduct")[0].tolist()
    gks = sorted(list(zip(ranks, persona)), key=lambda x: x[0], reverse=True)
    print("знания о персоне:", gks)
    gks = [gk[1] for gk in gks[:2]]

    # generate
    dict_inp = [{"context": context, "gk": gks, "candidate": ""}]
    gpt_inp = gpt_callator.test(dict_inp)[0]["input_ids"][:, :-1]
    len_gpt_inp = gpt_inp.size()[-1]
    print(("         (" + gpt_tokenizer.batch_decode(gpt_inp)[0] + ")"))
    gpt_out = generative_model.GPT.generate(
        gpt_inp,
        # do_sample=True,
        max_new_tokens=32,
        # pad_token_id=gpt_tokenizer.eos_token_id,
        # eos_token_id=gpt_tokenizer.eos_token_id,
        # num_beams=10,
        # temperature=1.0,
        # length_penalty=1,
        # no_repeat_ngram_size=3,
    )
    gpt_answer = gpt_out
    answer_raw = gpt_tokenizer.decode(gpt_answer[0], skip_special_tokens=False)
    print(("         (" + answer_raw + ")"))

    # proc answer
    answer = gpt_tokenizer.decode(
        gpt_answer[0, len_gpt_inp + 1 :], skip_special_tokens=True
    )
    context.append(answer)
    print("model:", answer)
