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
        scheduler_len,
        num_warmup_steps,
        lr,
        batch_size,
    ):
        super().__init__()
        self.transformer = transformer
        self.tokenizers = tokenizer
        self.scheduler_len = scheduler_len
        self.num_warmup_steps = num_warmup_steps
        self.lr = lr
        self.batch_sizes = batch_size

        # BiEncoder
        self.bienc_answer_loss = torch.nn.CrossEntropyLoss()
        self.bienc_gk_loss = torch.nn.CrossEntropyLoss()
        bienc_metrics = torchmetrics.MetricCollection(
            {
                f"r1@{self.batch_size}": torchmetrics.RetrievalRecall(k=1),
                f"r5@{self.batch_size}": torchmetrics.RetrievalRecall(k=5),
                f"mrr@{self.batch_size}": torchmetrics.RetrievalMRR(),
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

    def bi_encode(self, batch):
        """encode query or candidate for bi-encoder with use of encoder part of t5

        Args:
            batch (_type_): _description_

        Returns:
            _type_: _description_
        """
        batch = self.transformer(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["input_ids"],
            output_hidden_states=True,
        )
        batch = batch.encoder_last_hidden_state[:, 0, :]
        return batch

    def compute_sim(self, x, y, sim_func="cossim"):
        if sim_func == "cossim":
            x = x / x.norm(dim=1)[:, None]
            y = y / y.norm(dim=1)[:, None]
            sim = torch.mm(x, y.transpose(0, 1)) * 10
        elif sim_func == "dotprod":
            sim = torch.mm(x, y.transpose(0, 1))
        return sim

    def BiEnc_rank(self, batch):
        for k in batch:
            if k == "context":
                query = self.BiEncode(batch[k])
            elif k == "answer":
                candidat = self.BiEncode(batch[k])
                loss_f = self.BiEnc_answer_loss
            elif k == "gk":
                candidat = self.BiEncode(batch[k])
                loss_f = self.BiEnc_gk_loss

        sim = self.compute_sim(query, candidat)
        b_size = query.size()[0]
        labels = torch.zeros((b_size, b_size), dtype=torch.long, device=self.device)
        labels.fill_diagonal_(1)
        loss = loss_f(sim, torch.argmax(labels, 1))
        preds = sim.view(-1)
        targets = labels.view(-1)
        indexes = (
            torch.arange(sim.shape[0]).unsqueeze(1).expand_as(sim).reshape(preds.shape)
        )

        return {
            "loss": loss,
            "preds": preds,
            "targets": targets,
            "indexes": indexes,
            "b_size": b_size,
        }

    def training_step(self, batch, batch_idx):
        self.training_step_id = batch_idx
        batch_type = batch["type"]
        batch = batch["batch"]
        if batch_type == "answer":
            BiEnc_answer_out = self.BiEnc_rank(batch)
            loss = BiEnc_answer_out["loss"]

            # log answer
            self.log(
                "train_BiEnc_answer_loss",
                BiEnc_answer_out["loss"],
                on_epoch=True,
                on_step=True,
                prog_bar=True,
            )
            m = self.train_BiEnc_answer_metrics(
                BiEnc_answer_out["preds"],
                BiEnc_answer_out["targets"],
                indexes=BiEnc_answer_out["indexes"],
            )
            self.log_dict(
                m,
                on_epoch=True,
                on_step=True,
            )
        elif batch_type == "gk":
            BiEnc_gk_out = self.BiEnc_rank(batch)
            loss = BiEnc_gk_out["loss"]
            lr = BiEnc_gk_out["lr"]
            # log gk
            self.log(
                "train_BiEnc_gk_loss",
                BiEnc_gk_out["loss"],
                on_epoch=True,
                on_step=False,
                prog_bar=True,
                # batch_size=BiEnc_gk_out["b_size"],
            )
            m = self.train_BiEnc_gk_metrics(
                BiEnc_gk_out["preds"],
                BiEnc_gk_out["targets"],
                indexes=BiEnc_gk_out["indexes"],
            )
            self.log_dict(
                m,
                on_epoch=True,
                on_step=False,
                # batch_size=BiEnc_gk_out["b_size"],
            )
        lr = self.trainer.optimizers[0].param_groups[0]["lr"]
        # log all
        self.log(
            "lr",
            lr,
            on_epoch=False,
            on_step=True,
            prog_bar=True,
        )
        self.log(
            "train_loss",
            loss,
            on_epoch=True,
            on_step=True,
            prog_bar=True,
        )

        return loss

    def validation_step(self, batch, batch_idx):
        dataloader_idx = batch[1]
        batch = batch[0]
        if dataloader_idx == 0:
            BiEnc_answer_out = self.BiEnc_rank(batch)
            # log answer
            self.log(
                "val_BiEnc_answer_loss",
                BiEnc_answer_out["loss"],
                on_epoch=True,
                on_step=True,
                prog_bar=True,
            )
            m = self.val_BiEnc_answer_metrics(
                BiEnc_answer_out["preds"],
                BiEnc_answer_out["targets"],
                indexes=BiEnc_answer_out["indexes"],
            )
            self.log_dict(
                m,
                on_epoch=True,
                on_step=True,
            )
        elif dataloader_idx == 1:
            BiEnc_gk_out = self.BiEnc_rank(batch)
            # log gk
            self.log(
                "val_BiEnc_gk_loss",
                BiEnc_gk_out["loss"],
                on_epoch=True,
                on_step=True,
                prog_bar=True,
            )
            m = self.val_BiEnc_gk_metrics(
                BiEnc_gk_out["preds"],
                BiEnc_gk_out["targets"],
                indexes=BiEnc_gk_out["indexes"],
            )
            self.log_dict(
                m,
                on_epoch=True,
                on_step=True,
            )
