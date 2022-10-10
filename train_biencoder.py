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

pl.utilities.seed.seed_everything(42)

os.environ["http_proxy"] = "http://proxy.ad.speechpro.com:3128"
os.environ["https_proxy"] = "http://proxy.ad.speechpro.com:3128"
os.environ["ftp_proxy"] = "http://proxy.ad.speechpro.com:3128"

# config bert
parser = argparse.ArgumentParser()
bert_args = parser.parse_args("")
with open("configs/bert_config.json", "r") as config:
    opt = json.load(config)
vars(bert_args).update(opt)

# bert tokenizer
with open(bert_args.special_tokens_dict, "r") as config:
    special_tokens_dict = json.load(config)

bert_tokenizer = transformers.AutoTokenizer.from_pretrained(
    bert_args.pretrained_bert,
    truncation_side=bert_args.truncation_side,
    padding_side=bert_args.padding_side,
)
bert_tokenizer.add_special_tokens(special_tokens_dict)

# bert
context_bert = transformers.AutoModel.from_pretrained(bert_args.pretrained_bert)
context_bert.resize_token_embeddings(len(bert_tokenizer))
candidate_bert = transformers.AutoModel.from_pretrained(bert_args.pretrained_bert)
candidate_bert.resize_token_embeddings(len(bert_tokenizer))

# dataset
dataset = PersonaDataset(
    bert_args.data_path, mod="get_examples_gk", rnd_context=bert_args.rnd_context
)
train_size = len(dataset) - len(dataset) // bert_args.val_split
val_size = len(dataset) // bert_args.val_split
vars(bert_args).update({"train_size": train_size, "val_size": val_size})
train_dataset, val_dataset = torch.utils.data.random_split(
    dataset, [train_size, val_size]
)

# bert_callator
bert_callator = RetrievalCollator(
    bert_tokenizer, padding=bert_args.padding, max_length=bert_args.context_len
)

# dataloader
train_dataloader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=bert_args.batch_size,
    shuffle=True,
    collate_fn=bert_callator,
)
val_dataloader = torch.utils.data.DataLoader(
    val_dataset,
    batch_size=bert_args.batch_size,
    shuffle=False,
    collate_fn=bert_callator,
)

# scheduler len
scheduler_len = len(train_dataloader) * bert_args.epochs

# pl model
model = RetrievalModel(
    context_bert,
    candidate_bert,
    bert_args.batch_size,
    scheduler_len,
    bert_args.num_warmup_steps,
    bert_args.lr,
    aggregation_mod=bert_args.aggregation_mod,
    sim_mod=bert_args.sim_mod,
)

# logger
logger = pl.loggers.comet.CometLogger(
    api_key=bert_args.api_key,
    save_dir=bert_args.save_dir,
    project_name=bert_args.project_name,
    experiment_name=bert_args.experiment_name,
)
logger.log_hyperparams(bert_args)

# trainer
trainer = pl.Trainer(
    max_epochs=bert_args.epochs,
    accelerator="gpu",
    devices=1,
    gradient_clip_val=bert_args.gradient_clip_val,
    logger=logger,
    num_sanity_val_steps=1,
)

# fit
trainer.fit(model, train_dataloader, val_dataloader)
