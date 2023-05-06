from typing import *

import pytorch_lightning as pl
import transformers
import datasets

from model import *
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

        self.train = None
        self.val = None
        self.state = None
        self.skip_train_batches = 0
        self.skip_val_batches = 0
        self.epoch_samples_dropped = 0
        self.tokenizer = tokenizer
        self.data_dir = data_dir

        self.tokenizer.add_special_tokens(
            {
                "additional_special_tokens": parse_recursive_dict(
                    self.hparams.spec_tokens
                )
            }
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
                        "bs":{"train":task.train_bs}
                    }
                )

    def train_dataloader(self):
        batch_sizes = []
        for task in self.tasks:
            for dataset_name in task.datasets:
                batch_sizes.append(task.train_bs)
        try:
            ep = self.trainer.current_epoch
        except:
            ep = 0
        #shuffle train split
        datasets = [
            {
                "task_name": ds_dict["task_name"],
                "ds_name": ds_dict["ds_name"],
                "dataset": ds_dict["dataset"]["train"].shuffle(
                    seed=self.hparams.seed + ep
                ),
            }
            for ds_dict in self.datasets
        ]
        #make batch
        datasets = [
            ds["dataset"].map(
                lambda batch, task_name, ds_name: {
                    batch: self.multi_collator(task_name, ds_name, batch)
                },
                batched=True,
                batch_size=self.hparams.train_batch_size,
                remove_columns=ds.column_names,
                fn_kwargs={"task": ds["task_name"], "ds_name": ds["ds_name"]},
                drop_last_batch=False,
                num_proc=1,
            )
            for ds in self.datasets
        ]
        datasets = []
        ds_num_batch = []
        for ds in self.datasets:


        
        ds_num_batch = [len(ds) for ds in datasets]
        ds_prob = [size / sum(ds_num_batch) for size in ds_num_batch]
        train_dataloader = datasets.interleave_datasets(datasets, probabilities=ds_prob)
        return train_dataloader.with_format("pytorch")

    def val_dataloader(self):
        return None

    def test_dataloader(self):
        return None
