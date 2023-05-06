from typing import *

import pytorch_lightning as pl
import torch
import torchmetrics
import transformers
import pandas as pd
import numpy as np
import os
from data import MultiDataModule
from utils import *

import requests


class MultitaskModel(pl.LightningModule):
    def __init__(
        self,
        model_name_or_path: str,
        data_dir: str,
        spec_tokens_dict: Dict,
        tasks: Dict,
        seed: int = 42,
        lr: float = 5e-05,
        num_warmup_steps: int = 100,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.transformer = transformers.T5ForConditionalGeneration.from_pretrained(
            self.hparams.model_name_or_path
        )
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(model_name_or_path)
        # init tasks
        self.metrics_dict = {}
        task_list = []
        for task_name in tasks:
            task = TaskConfig(
                task_name=task_name,
                tokenizer=self.tokenizer,
                spec_tokens_dict=spec_tokens_dict,
                **tasks[task_name],
            )
            task_list.append(task)
            # init metrics
            # if task.collator_type == "multiclass":
            #     metrics = torchmetrics.MetricCollection(
            #         {
            #             f"{task.task_name}_F1": torchmetrics.F1Score(
            #                 task="multiclass", num_classes=len(task.collator.classes)
            #             )
            #         }
            #     )
            # elif task.collator_type == "multilabel":
            #     metrics = torchmetrics.MetricCollection(
            #         {
            #             f"{task.task_name}_F1": torchmetrics.F1Score(
            #                 task="multilabel", num_labels=len(task.collator.classes)
            #             )
            #         }
            #     )
            # elif task.collator_type == "knowledgegrounded":
            #     metrics = torchmetrics.MetricCollection(
            #         {
            #             f"{task.task_name}_BLEU1": torchmetrics.BLEUScore(n_gram=1),
            #             f"{task.task_name}_BLEU2": torchmetrics.BLEUScore(n_gram=2),
            #         }
            #     )
            # elif task.collator_type == "crossencoder":
            #     metrics = torchmetrics.MetricCollection(
            #         {
            #             f"{task.task_name}_Recall": torchmetrics.Recall(
            #                 task="multiclass", num_classes=len(task.collator.scores)
            #             ),
            #         }
            #     )
            # self.metrics_dict[task.task_name] = {}
            # self.metrics_dict[task.task_name]["train"] = metrics.clone(prefix="train_")
            # self.metrics_dict[task.task_name]["val"] = metrics.clone(prefix="val_")

        # init data module
        self.data_module = MultiDataModule(
            tasks=task_list,
            tokenizer=self.tokenizer,
            data_dir=self.hparams.data_dir,
            spec_tokens=self.hparams.spec_tokens_dict,
            seed=self.hparams.seed,
        )

    def seq2seq(self, batch, meta):
        loss = self.transformer(**batch["inp"], labels=batch["out"]["input_ids"]).loss
        out = self.transformer.generate(
            **batch["inp"], max_length=batch["out"]["input_ids"].size()[-1] + 1
        )
        pred = self.data_module.multi_collator.label_decode(batch=out, **meta)
        target = self.data_module.multi_collator.label_decode(
            batch=batch["out"]["input_ids"], **meta
        )
        # print(meta)
        # print("pred")
        # print(pred)
        # print("target")
        # print(target)
        return loss, pred, target

    def forward(self, batch, meta):
        if meta["forward_type"] == "seq2seq":
            loss, pred, target = self.seq2seq(batch, meta)

        return loss, pred, target

    def training_step(self, batch: dict, batch_idx):
        loss, pred, target = self(**batch)
        metrics = self.metrics_dict[batch["meta"]["task_name"]]["train"](pred, target)
        # Log
        self.log("train_loss", loss, sync_dist=True)
        self.log_dict(metrics, sync_dist=True)

        return loss

    def training_epoch_end(self, outputs):
        for task_name in self.metrics_dict:
            for split in self.metrics_dict[task_name]:
                self.metrics_dict[task_name][split].reset()

    def validation_step(self, batch: dict, batch_idx):
        loss, pred, target = self(**batch)
        metrics = self.metrics_dict[batch["meta"]["task_name"]]["val"](pred, target)

        detection_tp = 0.0
        detection_fn = 0.0
        detection_fp = 0.0

        selection_total = 0.0
        selection_tp = 0.0
        selection_fp = 0.0
        selection_fn = 0.0
        selection_exact_matched = 0.0

        rouge1_sum = 0.0
        rouge2_sum = 0.0
        rougeL_sum = 0.0

        ref_responses = []
        pred_responses = []

        for t, p in zip(target, pred):
            if batch["meta"]["task_name"] == "Knowledge_Seeking":
                if t.item() == 1:
                    if p.item() == 1:
                        detection_tp += 1
                    else:
                        detection_fn += 1
                else:
                    if p.item() == 1:
                        detection_fp += 1

            elif batch["meta"]["task_name"] == "Entity_Detection":
                if list(t).count(1) > 0 or list(p).count(1) > 0:
                    selection_total += 1.0
                    num_matched = 0
                    for ind, ref in enumerate(list(t)):
                        if ref == 1 and ref == list(p)[ind]:
                            num_matched += 1
                    tp = num_matched
                    fp = list(p).count(1) - num_matched
                    fn = list(t).count(1) - num_matched
                    if list(p).count(1) == list(t).count(1) and list(t).count(1) == tp:
                        exact_matched = 1
                    else:
                        exact_matched = 0
                    selection_tp += float(tp)
                    selection_fp += float(fp)
                    selection_fn += float(fn)
                    selection_exact_matched += float(exact_matched)

            elif batch["meta"]["task_name"] == "Response_Generation":
                ref_responses.append(t)
                pred_responses.append(p)
                scorer = rouge_scorer.RougeScorer(
                    ["rouge1", "rouge2", "rougeL"], use_stemmer=True
                )
                scores = scorer.score(t, p)
                rouge1 = scores["rouge1"].fmeasure
                rouge2 = scores["rouge2"].fmeasure
                rougeL = scores["rougeL"].fmeasure
                rouge1_sum += rouge1
                rouge2_sum += rouge2
                rougeL_sum += rougeL

        dstc_metrics = {
            "detection_tp": detection_tp,
            "detection_fp": detection_fp,
            "detection_fn": detection_fn,
            "selection_total": selection_total,
            "selection_tp": selection_tp,
            "selection_fp": selection_fp,
            "selection_fn": selection_fn,
            "selection_exact_matched": selection_exact_matched,
            "rouge1": rouge1_sum,
            "rouge2": rouge2_sum,
            "rougeL": rougeL_sum,
            "ref_responses": ref_responses,
            "pred_responses": pred_responses,
        }

        # Log
        self.log("val_loss", loss, sync_dist=True)
        self.log_dict(metrics, sync_dist=True)

        return dstc_metrics

    def validation_epoch_end(self, outputs):
        detection_tp = 0.0
        detection_fp = 0.0
        detection_fn = 0.0

        selection_total = 0.0
        selection_tp = 0.0
        selection_fp = 0.0
        selection_fn = 0.0
        selection_exact_matched = 0.0

        rouge1_sum = 0.0
        rouge2_sum = 0.0
        rougeL_sum = 0.0

        ref_responses = []
        pred_responses = []

        det_precision = 0.0
        det_recall = 0.0

        for output in outputs:
            if type(output) == dict:
                detection_tp += output["detection_tp"]
                detection_fp += output["detection_fp"]
                detection_fn += output["detection_fn"]

                selection_total += output["selection_total"]
                selection_tp += output["detection_tp"]
                selection_fp += output["selection_fp"]
                selection_fn += output["selection_fn"]
                selection_exact_matched += output["selection_exact_matched"]

                rouge1_sum += output["rouge1"]
                rouge2_sum += output["rouge2"]
                rougeL_sum += output["rougeL"]

                ref_responses.extend(output["ref_responses"])
                pred_responses.extend(output["pred_responses"])

        # selection
        if selection_tp + selection_fp > 0:
            selection_p = selection_tp / (selection_tp + selection_fp)
        else:
            selection_p = 0.0

        if selection_tp + selection_fn > 0:
            selection_r = selection_tp / (selection_tp + selection_fn)
        else:
            selection_r = 0.0

        if selection_p + selection_r > 0.0:
            selection_f = 2 * selection_p * selection_r / (selection_p + selection_r)
        else:
            selection_f = 0.0
        selection_em_acc = selection_exact_matched / selection_total

        # BLEU
        bleu_metric = BleuMetric()
        bleu_score = (
            bleu_metric.evaluate_batch(pred_responses, ref_responses)["bleu"]
            / 100.0
            * detection_tp
        )

        # Meteor setting can be tagged or deleted after first launch
        meteor_file_path = summ_eval.__file__
        meteor_dir = os.path.dirname(meteor_file_path)
        if not os.path.exists(os.path.join(meteor_dir, "data")):
            os.mkdir(os.path.join(meteor_dir, "data"))
        if not os.path.exists(os.path.join(meteor_dir, "data", "paraphrase-en.gz")):
            paraphrase_en_gz_url = "https://github.com/tylin/coco-caption/blob/master/pycocoevalcap/meteor/data/paraphrase-en.gz?raw=true"
            r = requests.get(paraphrase_en_gz_url)
            with open(
                os.path.join(meteor_dir, "data", "paraphrase-en.gz"), "wb"
            ) as outputf:
                outputf.write(r.content)
        # Meteor metric starts here
        meteor_metric = MeteorMetric()
        meteor_score = (
            meteor_metric.evaluate_batch(pred_responses, ref_responses)["meteor"]
            * detection_tp
        )

        summed_scores = [
            detection_tp,
            rouge1_sum,
            rouge2_sum,
            rougeL_sum,
            bleu_score,
            meteor_score,
        ]

        f_scores = []
        for score in summed_scores:
            if detection_tp + detection_fp > 0.0:
                scoreP = score / (detection_tp + detection_fp)
            else:
                scoreP = 0.0
            if detection_tp + detection_fn > 0.0:
                scoreR = score / (detection_tp + detection_fn)
            else:
                scoreR = 0.0
            if scoreP + scoreR > 0.0:
                score_f = 2 * scoreP * scoreR / (scoreP + scoreR)
            else:
                score_f = 0.0
            if score == summed_scores[0]:
                det_precision = scoreP
                det_recall = scoreR
            f_scores.append(score_f)

        self.log("DSTC_detection_precision", det_precision, sync_dist=True)
        self.log("DSTC_detection_recall", det_recall, sync_dist=True)
        self.log("DSTC_detection_f1score", f_scores[0], sync_dist=True)

        self.log("DSTC_selection_precision", selection_p, sync_dist=True)
        self.log("DSTC_selection_recall", selection_r, sync_dist=True)
        self.log("DSTC_selection_f1score", selection_f, sync_dist=True)
        self.log("DSTC_selection_em_acc", selection_em_acc, sync_dist=True)

        self.log("DSTC_rouge1", f_scores[1], sync_dist=True)
        self.log("DSTC_rouge2", f_scores[2], sync_dist=True)
        self.log("DSTC_rougeL", f_scores[3], sync_dist=True)

        self.log("DSTC_BLEU", f_scores[4], sync_dist=True)
        self.log("DSTC_METEOR", f_scores[5], sync_dist=True)

        for task_name in self.metrics_dict:
            for split in self.metrics_dict[task_name]:
                self.metrics_dict[task_name][split].reset()

    def test_step(self, batch: dict, batch_idx):
        # Compute embeddings
        embeddings_a = self(
            [
                batch["batch"]["query"]["input_ids"],
                batch["batch"]["query"]["attention_mask"],
            ]
        )
        embeddings_b = self(
            [
                batch["batch"]["candidate"]["input_ids"],
                batch["batch"]["candidate"]["attention_mask"],
            ]
        )

        labels = batch["batch"]["labels"]

        return {
            "type": batch["type"],
            "query_embs": embeddings_a.cpu(),
            "candidate_embs": embeddings_b.cpu(),
            "labels": labels.cpu(),
        }

    def test_epoch_end(self, outputs: Union[dict, List[dict]]) -> None:
        outputs = self.join_steps_outputs(outputs)
        all_metrics = {}
        losses = []
        for task in outputs.keys():
            for source in outputs[task]:
                scores = (
                    torch.mm(
                        outputs[task][source]["query_embs"],
                        outputs[task][source]["candidate_embs"].transpose(0, 1),
                    )
                    * self.hparams.scale
                )

                labels = outputs[task][source]["labels"]
                loss = self.clip_loss(scores, labels)

                # Calculate metrics
                preds = scores.view(-1).cpu()
                targets = labels.reshape(preds.shape)
                indexes = (
                    torch.arange(scores.shape[0])
                    .unsqueeze(1)
                    .expand_as(scores)
                    .reshape(preds.shape)
                )

                test_metrics = self.make_metrics_collection(
                    "test", scores.shape[0], task, source
                )
                metrics = test_metrics(preds, targets, indexes)
                all_metrics = all_metrics | metrics
                losses.append(loss)
                # Log
                self.log("loss", loss, sync_dist=True)
                self.log_dict(metrics, sync_dist=True)

        all_metrics = {k: all_metrics[k].item() for k in all_metrics}
        df = pd.DataFrame(all_metrics, index=[0])
        df["average"] = df.mean(numeric_only=True, axis=1)
        df["avrg_loss"] = sum(losses) / len(losses)
        df = df.round(4)
        df.to_csv(
            os.path.join(self.trainer.logger.log_dir, "test_metrics.csv"), index=False
        )

        print(f"Results were saved to {self.trainer.logger.log_dir}/test_metrics.csv")

        return super().test_epoch_end(outputs)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)
        scheduler = transformers.get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.hparams.num_warmup_steps,
            num_training_steps=self.trainer.estimated_stepping_batches,
        )
        return [optimizer], [
            {"scheduler": scheduler, "name": "cosine_scheduler", "interval": "step"}
        ]
