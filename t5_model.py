import torch
import pytorch_lightning as pl
import torchmetrics
import transformers


class single_multitask_model(pl.LightningModule):
    """_summary_

    Attributes:
        transformer (): model
        tokenizers (): tokenizer
        scheduler_len (int): len dataloder * num epochs
        num_warmup_steps (int): len of warmup
        lr (int): learning rate
        batch_size (int): size of batch ?единый для всех тасков?
    """

    def __init__(
        self,
        transformer,
        tokenizer,
        scheduler_len: int,
        num_warmup_steps: int,
        lr: float,
    ):
        super().__init__()
        self.transformer = transformer
        self.tokenizers = tokenizer
        self.scheduler_len = scheduler_len
        self.num_warmup_steps = num_warmup_steps
        self.lr = lr

        # BiEncoder
        self.bienc_answer_loss = torch.nn.CrossEntropyLoss()
        self.bienc_gk_loss = torch.nn.CrossEntropyLoss()
        bienc_metrics = torchmetrics.MetricCollection(
            {
                "r1": torchmetrics.RetrievalRecall(k=1),
                "r5": torchmetrics.RetrievalRecall(k=5),
                "mrr": torchmetrics.RetrievalMRR(),
            }
        )
        self.train_bienc_answer_metrics = bienc_metrics.clone(prefix="train_answer_")
        self.train_bienc_gk_metrics = bienc_metrics.clone(prefix="train_gk_")
        self.val_bienc_answer_metrics = bienc_metrics.clone(prefix="val_answer_")
        self.val_bienc_gk_metrics = bienc_metrics.clone(prefix="val_gk_")

        # Generative
        self.gen_metrics = torchmetrics.MetricCollection(
            {
                "val_BLEU1": torchmetrics.BLEUScore(n_gram=1),
                "val_BLEU2": torchmetrics.BLEUScore(n_gram=2),
                "val_BLEU4": torchmetrics.BLEUScore(n_gram=4),
            }
        )

        self.save_hyperparameters(ignore=["transformer"])

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        scheduler = transformers.get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.num_warmup_steps,
            num_training_steps=self.scheduler_len,
        )
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]

    def bi_encode(self, batch: dict):
        batch = self.transformer(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["input_ids"],
            output_hidden_states=True,  # remove for enc only
        )
        batch = batch.encoder_last_hidden_state[
            :, 0, :
        ]  # last_hidden_state for enc only
        return batch

    def compute_sim(self, x: torch.tensor, y: torch.tensor, sim_func: str = "cossim"):
        if sim_func == "cossim":
            x = x / x.norm(dim=1)[:, None]
            y = y / y.norm(dim=1)[:, None]
            sim = torch.mm(x, y.transpose(0, 1)) * 10
        elif sim_func == "dotprod":
            sim = torch.mm(x, y.transpose(0, 1))
        return sim

    def bienc_rank(self, batch, loss_f, metrics):
        query = self.bi_encode(batch["query"])
        candidat = self.bi_encode(batch["candidat"])
        # loss
        sim = self.compute_sim(query, candidat)
        b_size = candidat.size()[0]
        labels = torch.zeros((b_size, b_size), dtype=torch.long, device=self.device)
        labels.fill_diagonal_(1)
        loss = loss_f(sim, torch.argmax(labels, 1))
        # metrics
        preds = sim.view(-1)
        targets = labels.view(-1)
        indexes = (
            torch.arange(sim.shape[0]).unsqueeze(1).expand_as(sim).reshape(preds.shape)
        )
        metrics = metrics(preds, targets, indexes)
        metrics = {k + f"@{b_size}": metrics[k] for k in metrics}

        return {"loss_": loss, "metrics": metrics}

    def training_step(self, batch, batch_idx):
        batch_type = batch["type"]
        batch = batch["batch"]
        # training
        if batch_type == "answer":
            out = self.bienc_rank(
                batch,
                self.bienc_answer_loss,
                self.train_bienc_answer_metrics,
            )
        elif batch_type == "gk":
            out = self.bienc_rank(
                batch,
                self.bienc_gk_loss,
                self.train_bienc_gk_metrics,
            )
        # logging
        self.log(
            "lr",
            self.trainer.optimizers[0].param_groups[0]["lr"],
            on_epoch=False,
            on_step=True,
        )
        self.log(
            "train_loss",
            out["loss_"],
            on_epoch=True,
            on_step=True,
        )
        self.log(
            f"train_{batch_type}_loss",
            out["loss_"],
            on_epoch=True,
            on_step=True,
        )
        self.log_dict(
            out["metrics"],
            on_epoch=True,
            on_step=True,
        )
        return out["loss_"]

    def validation_step(self, batch, batch_idx):
        batch_type = batch["type"]
        batch = batch["batch"]
        # validation
        if batch_type == "answer":
            out = self.bienc_rank(
                batch,
                self.bienc_answer_loss,
                self.val_bienc_answer_metrics,
            )
        elif batch_type == "gk":
            out = self.bienc_rank(
                batch,
                self.bienc_gk_loss,
                self.val_bienc_gk_metrics,
            )
        # logging
        self.log(
            "val_loss",
            out["loss_"],
            on_epoch=True,
            on_step=False,
        )
        self.log(
            f"val_{batch_type}_loss",
            out["loss_"],
            on_epoch=True,
            on_step=False,
        )
        self.log_dict(
            out["metrics"],
            on_epoch=True,
            on_step=False,
        )
