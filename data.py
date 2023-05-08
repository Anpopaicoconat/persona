from typing import *
import os

import pytorch_lightning as pl
import transformers
import datasets

# from model import *
from utils import *


def parse_recursive_dict(inp_dict, tokens=None):
    tokens = tokens or []
    for k in inp_dict:
        if isinstance(inp_dict[k], dict):
            tokens = parse_recursive_dict(inp_dict[k], tokens=tokens)
        else:
            tokens.append(inp_dict[k])
    return tokens


class MultiDataModule(pl.LightningDataModule):
    def __init__(
        self,
        tasks: List,
        tokenizer: transformers.AutoTokenizer,
        data_dir: str = "",
        spec_tokens: Dict = {},
        seed: int = 42,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.skip_train_batches = 0
        self.skip_val_batches = 0
        self.epoch_samples_dropped = 0
        self.tokenizer = tokenizer
        self.data_dir = data_dir
        self.seed = seed

        self.tokenizer.add_special_tokens(
            {"additional_special_tokens": parse_recursive_dict(spec_tokens)}
        )
        self.tasks = tasks
        self.multi_collator = MultiCollator(
            {task.task_name: task.collator for task in self.tasks}
        )
        self.datasets = []
        for task in self.tasks:
            for dataset_name in task.datasets:
                self.datasets.append(
                    {
                        "task_name": task.task_name,
                        "ds_name": dataset_name,
                        "dataset": datasets.load_from_disk(
                            os.path.join(self.data_dir, task.datasets[dataset_name])
                        ),
                        "bs": {"train": task.train_bs, "val": task.val_bs},
                    }
                )

    def train_dataloader(self):
        try:
            ep = self.trainer.current_epoch
        except:
            ep = 0

        dataset_list = []
        ds_prob = []
        for ds_dict in self.datasets:
            ds = ds_dict["dataset"]["train"].shuffle(seed=self.seed + ep)
            ds = ds.map(
                lambda batch, task_name, ds_name: self.multi_collator(
                    task_name, ds_name, batch
                ),
                batched=True,
                batch_size=ds_dict["bs"]["train"],
                remove_columns=ds.column_names,
                fn_kwargs={
                    "task_name": ds_dict["task_name"],
                    "ds_name": ds_dict["ds_name"],
                },
                drop_last_batch=False,
                num_proc=1,
                load_from_cache_file=True,
            )
            dataset_list.append(ds)
            ds_prob.append(len(ds))
        ds_prob = [l / sum(ds_prob) for l in ds_prob]
        ds_prob = [1 / len(dataset_list) for _ in dataset_list]
        train_dataloader = datasets.interleave_datasets(
            dataset_list, probabilities=ds_prob
        )
        return train_dataloader.with_format("pytorch")
