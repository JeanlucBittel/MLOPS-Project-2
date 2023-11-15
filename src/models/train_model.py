import os
import datasets
import torch
import wandb
from dotenv import load_dotenv
from datetime import datetime
from typing import Optional
from pytorch_lightning import LightningModule
from transformers import AutoConfig, get_linear_schedule_with_warmup, AutoModelForSequenceClassification

class GLUETransformer(LightningModule):
    def __init__(
        self,
        model_name_or_path: str,
        num_labels: int,
        task_name: str,
        learning_rate: float = 2e-5,
        adam_epsilon: float = 1e-8,
        warmup_steps: int = 0,
        weight_decay: float = 0.0,
        train_batch_size: int = 32,
        eval_batch_size: int = 32,
        eval_splits: Optional[list] = None,
        **kwargs,
    ):
        super().__init__()

        self.save_hyperparameters()

        self.config = AutoConfig.from_pretrained(model_name_or_path, num_labels=num_labels)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path, config=self.config)
        self.metric = datasets.load_metric(
            "glue", self.hparams.task_name, experiment_id=datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
        )

        self.training_step_outputs = []
        self.val_step_outputs = []

        load_dotenv()
        self.WANDB_API_KEY = os.getenv('WANDB_API_KEY')

        if self.WANDB_API_KEY:
            wandb.init(
                # Set the project where this run will be logged
                project="MLOPS",
                # We pass a run name (otherwise it’ll be randomly assigned, like sunshine-lollypop-10)
                # name=f"{self.wandb_name}_epoch_{self.epoch}",
                reinit=True,
                # Track hyperparameters and run metadata
                config={
                    "architecture": "DistilBERT",
                    "dataset": "MRPC",
                    "learning_rate": self.hparams.learning_rate,
                    "adam_epsilon": self.hparams.adam_epsilon,
                    "warmup_steps": self.hparams.warmup_steps,
                    "weight_decay": self.hparams.weight_decay,
                    "train_batch_size": self.hparams.train_batch_size,
                    "eval_batch_size": self.hparams.eval_batch_size
                })

            wandb.log({
                "learning_rate": self.hparams.learning_rate,
                "adam_epsilon": self.hparams.adam_epsilon,
                "warmup_steps": self.hparams.warmup_steps,
                "weight_decay": self.hparams.weight_decay,
                "train_batch_size": self.hparams.train_batch_size,
                "eval_batch_size": self.hparams.eval_batch_size
            })

    def forward(self, **inputs):
        return self.model(**inputs)

    def training_step(self, batch, batch_idx):
        outputs = self(**batch)
        loss = outputs[0]

        if self.WANDB_API_KEY:
            wandb.log({
                "learning_rate": self.hparams.learning_rate,
                "adam_epsilon": self.hparams.adam_epsilon,
                "warmup_steps": self.hparams.warmup_steps,
                "weight_decay": self.hparams.weight_decay,
                "train_batch_size": self.hparams.train_batch_size,
                "eval_batch_size": self.hparams.eval_batch_size
            })
            metrics = {
                "train/train_loss": loss
            }
            wandb.log(metrics)

        self.training_step_outputs.append(loss)
        self.log("train/train_loss", loss)

        return loss

    def on_train_epoch_end(self):
        loss = torch.stack(self.training_step_outputs).mean()

        if self.WANDB_API_KEY:
            metrics = {
                "train/average_train_loss": loss
            }
            wandb.log(metrics)

        self.log("train/train_average_loss", loss)

        self.training_step_outputs.clear()

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        outputs = self(**batch)
        val_loss, logits = outputs[:2]

        if self.hparams.num_labels > 1:
            preds = torch.argmax(logits, axis=1)
        elif self.hparams.num_labels == 1:
            preds = logits.squeeze()

        labels = batch["labels"]

        if self.WANDB_API_KEY:
            metrics = {
                "val/val_loss": val_loss
            }
            wandb.log(metrics)

        self.val_step_outputs.append(val_loss)
        self.log("val/val_loss", val_loss)

        return {"loss": val_loss, "preds": preds, "labels": labels}

    def on_validation_epoch_end(self):
        loss = torch.stack(self.val_step_outputs).mean()

        if self.WANDB_API_KEY:
            metrics = {
                "val/average_val_loss": loss
            }
            wandb.log(metrics)

        self.log("val/val_average_loss", loss)

        self.val_step_outputs.clear()

    def validation_epoch_end(self, outputs):
        if self.hparams.task_name == "mnli":
            for i, output in enumerate(outputs):
                # matched or mismatched
                split = self.hparams.eval_splits[i].split("_")[-1]
                preds = torch.cat([x["preds"] for x in output]).detach().cpu().numpy()
                labels = torch.cat([x["labels"] for x in output]).detach().cpu().numpy()
                loss = torch.stack([x["loss"] for x in output]).mean()
                self.log(f"val_loss_{split}", loss, prog_bar=True)
                split_metrics = {
                    f"{k}_{split}": v for k, v in self.metric.compute(predictions=preds, references=labels).items()
                }
                self.log_dict(split_metrics, prog_bar=True)
            return loss

        preds = torch.cat([x["preds"] for x in outputs]).detach().cpu().numpy()
        labels = torch.cat([x["labels"] for x in outputs]).detach().cpu().numpy()
        loss = torch.stack([x["loss"] for x in outputs]).mean()

        if self.WANDB_API_KEY:
            metrics = {
                "val/average_val_loss": loss
            }
            wandb.log(metrics)

        self.log("val_loss", loss, prog_bar=True)
        self.log_dict(self.metric.compute(predictions=preds, references=labels), prog_bar=True)

    def configure_optimizers(self):
        """Prepare optimizer and schedule (linear warmup and decay)"""
        model = self.model
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]

        optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=self.hparams.learning_rate, eps=self.hparams.adam_epsilon)

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.hparams.warmup_steps,
            num_training_steps=self.trainer.estimated_stepping_batches,
        )
        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}
        return [optimizer], [scheduler]
