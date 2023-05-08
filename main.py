from typing import *
import os
from pytorch_lightning.cli import LightningCLI
import pytorch_lightning as pl

import clearml
import pandas as pd

from model import MultitaskModel


class MultiTaskLightningCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        # Program-level
        parser.add_argument("--project_name", type=str, default="Persona")
        parser.add_argument(
            "--experiment_name",
            type=str,
            default="rut5-small",
        )
        parser.add_argument("--log_to_clearml", type=bool, default=True)
        parser.add_argument("--clearml_tags", type=list, default=[])
        parser.add_argument("--task_id", type=str, default="")
        parser.add_argument("--ckpt_path", type=str, default="")

    @staticmethod
    def activate_clearml(config, task_type="training"):
        if config["task_id"]:
            task = clearml.Task.init(
                reuse_last_task_id=config["task_id"],
                continue_last_task=0,
                task_type=task_type,
            )
            print(f"Reuse task: {task.id}")
        else:
            task = clearml.Task.init(
                project_name=config["project_name"],
                task_name=config["experiment_name"],
                task_type=task_type,
            )
            print(f"Create task: {task.id}")
        task.add_tags(config["clearml_tags"])

        return task


def main():
    cli = MultiTaskLightningCLI(
        MultitaskModel, save_config_kwargs={"overwrite": True}, run=False
    )

    if cli.config["log_to_clearml"] and cli.trainer.local_rank == 0:
        clearml_task = cli.activate_clearml(cli.config, task_type="training")
        cli.config["task_id"] = clearml_task.id
        config_dict = dict(cli.config)
        config_dict = {k.replace(".", "/"): config_dict[k] for k in config_dict}
        clearml_task.set_parameters(config_dict)

    cli.trainer.fit(
        model=cli.model,
        datamodule=cli.model.data_module,
        ckpt_path=cli.config["ckpt_path"],
    )


if __name__ == "__main__":
    main()
