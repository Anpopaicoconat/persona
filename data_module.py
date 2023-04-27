from typing import *

import pytorch_lightning as pl
import transformers

from model import *
from utils import InterleaveDatasetsLoader, MultiCollator


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
        shuffle_window_size: int = 64,
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
                        "path": task.datasets[dataset_name],
                    }
                )

    def train_dataloader(self):
        return None

    def val_dataloader(self):
        return None

    def test_dataloader(self):
        return None
