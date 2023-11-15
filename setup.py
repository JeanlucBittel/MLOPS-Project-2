import torch
from pytorch_lightning import seed_everything, Trainer
from src.data.make_dataset import GLUEDataModule
from src.models.train_model import GLUETransformer

def setup(batch_size, learning_rate, warmup_steps):
    #wandb.login()

    seed_everything(42)

    dm = GLUEDataModule(
        model_name_or_path="distilbert-base-uncased",
        task_name="mrpc",
        train_batch_size=batch_size,
        eval_batch_size=batch_size
    )
    dm.setup("fit")
    model = GLUETransformer(
        model_name_or_path="distilbert-base-uncased",
        num_labels=dm.num_labels,
        eval_splits=dm.eval_splits,
        task_name=dm.task_name,
        learning_rate=learning_rate,
        warmup_steps=warmup_steps,
        train_batch_size=batch_size,
        eval_batch_size=batch_size,
    )

    trainer = Trainer(
        max_epochs=3,
        accelerator="auto",
        devices=1 if torch.cuda.is_available() else None,
    )
    trainer.fit(model, datamodule=dm)

    # wandb.finish()