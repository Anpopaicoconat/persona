from typing import *
import os
from pytorch_lightning.cli import LightningCLI
import pytorch_lightning as pl

# from clearml import Task
import pandas as pd

from model import MultitaskModel


class MultiTaskLightningCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        # Program-level
        parser.add_argument("--project_name", type=str, default="Semantic Search")
        parser.add_argument(
            "--experiment_name",
            type=str,
            default="Train mpnet-paraphrase on paraphrase",
        )
        parser.add_argument("--log_to_clearml", type=bool, default=True)
        parser.add_argument("--clearml_tags", type=list, default=[])
        parser.add_argument("--task_id", type=str, default="")
        parser.add_argument("--ckpt_path", type=str, default="")
        parser.add_argument(
            "--clearml_api_host",
            type=str,
            default="http://nid-nlu-u20-clearml.ad.speechpro.com:8008",
        )
        parser.add_argument(
            "--clearml_web_host",
            type=str,
            default="http://nid-nlu-u20-clearml.ad.speechpro.com:8080",
        )
        parser.add_argument(
            "--clearml_files_host",
            type=str,
            default="http://nid-nlu-u20-clearml.ad.speechpro.com:8081",
        )

    @staticmethod
    def activate_clearml(config, task_type="training"):
        Task.set_credentials(
            api_host=config["clearml_api_host"],
            web_host=config["clearml_web_host"],
            files_host=config["clearml_files_host"],
        )

        if config["task_id"]:
            task = Task.init(
                reuse_last_task_id=config["task_id"],
                continue_last_task=0,
                task_type=task_type,
            )
            print(f"Reuse task: {task.id}")
        else:
            task = Task.init(
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

    # if cli.config["log_to_clearml"] and cli.trainer.local_rank == 0:
    #     clearml_task = cli.activate_clearml(cli.config, task_type="training")
    #     cli.config["task_id"] = clearml_task.id
    #     config_dict = dict(cli.config)
    #     config_dict = {k.replace(".", "/"): config_dict[k] for k in config_dict}
    #     clearml_task.set_parameters(config_dict)

    # cli.trainer.fit(
    #     model=cli.model,
    #     datamodule=cli.model.data_module,
    #     ckpt_path=cli.config["ckpt_path"],
    # )
    for batch in cli.model.data_module.train_dataloader():
        print(batch)
        0 / 0


if __name__ == "__main__":
    main()
