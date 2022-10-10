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

# gpt config
parser = argparse.ArgumentParser()
gpt_args = parser.parse_args("")
with open("configs/gpt_config.json", "r") as config:
    opt = json.load(config)
vars(gpt_args).update(opt)

# gpt tokenizer
with open(gpt_args.special_tokens_dict, "r") as config:
    special_tokens_dict = json.load(config)

gpt_tokenizer = transformers.AutoTokenizer.from_pretrained(
    gpt_args.pretrained_gpt,
    truncation_side=gpt_args.truncation_side,
    padding_side=gpt_args.padding_side,
)
gpt_tokenizer.pad_token = gpt_tokenizer.eos_token
gpt_tokenizer.add_special_tokens(special_tokens_dict)

# gpt
gpt = transformers.GPT2LMHeadModel.from_pretrained(gpt_args.pretrained_gpt)
gpt.resize_token_embeddings(len(gpt_tokenizer))

# dataset
dataset = PersonaDataset(
    gpt_args.data_path, mod="get_examples_gpt", rnd_context=gpt_args.rnd_context
)

train_size = len(dataset) - len(dataset) // gpt_args.val_split
val_size = len(dataset) // gpt_args.val_split
vars(gpt_args).update({"train_size": train_size, "val_size": val_size})

train_dataset, val_dataset = torch.utils.data.random_split(
    dataset, [train_size, val_size]
)

# gpt callator
gpt_callator = GenerativeCollator(
    gpt_tokenizer, padding=gpt_args.padding, max_length=gpt_args.max_len
)

# dataloader
train_dataloader = torch.utils.data.DataLoader(
    dataset, batch_size=gpt_args.batch_size, shuffle=True, collate_fn=gpt_callator
)
val_dataloader = torch.utils.data.DataLoader(
    val_dataset, batch_size=1, shuffle=False, collate_fn=gpt_callator.test
)

# scheduler len
scheduler_len = len(train_dataloader) * gpt_args.epochs

# pl model
model = GenerativeModel(
    gpt,
    gpt_tokenizer,
    gpt_args.batch_size,
    scheduler_len,
    gpt_args.num_warmup_steps,
    gpt_args.lr,
    gpt_args.max_len,
)

# logger
logger = pl.loggers.comet.CometLogger(
    api_key=gpt_args.api_key,
    save_dir=gpt_args.save_dir,
    project_name=gpt_args.project_name,
    experiment_name=gpt_args.experiment_name,
)
logger.log_hyperparams(gpt_args)

# checkpoint callback
checkpoint_callback = pl.callbacks.ModelCheckpoint(
    monitor="val_loss",
    dirpath=gpt_args.save_dir,
    filename="gpt-{epoch:02d}-{val_loss:.2f}",
    save_top_k=1,
    mode="min",
)

# trainer
trainer = pl.Trainer(
    max_epochs=gpt_args.epochs,
    accelerator="gpu",
    devices=1,
    gradient_clip_val=gpt_args.gradient_clip_val,
    logger=logger,
    num_sanity_val_steps=1,
    callbacks=[checkpoint_callback],
)

# fit
trainer.fit(model, train_dataloader, val_dataloader)
