# src/smthome/utils/callbacks.py

import pytorch_lightning as pl

class MetricsCallback(pl.Callback):
    def __init__(self):
        super().__init__()
        self.train_loss = []
        self.val_loss = []
        self.train_acc = []
        self.val_acc = []

    def on_train_epoch_end(self, trainer, pl_module):
        # Collect the training loss and accuracy after each epoch ends
        train_loss = trainer.callback_metrics.get("train_loss")
        train_acc = trainer.callback_metrics.get("train_acc")
        if train_loss is not None:
            self.train_loss.append(train_loss.item())
        if train_acc is not None:
            self.train_acc.append(train_acc.item())

    def on_validation_epoch_end(self, trainer, pl_module):
        # Collect the validation loss and accuracy after each epoch ends
        val_loss = trainer.callback_metrics.get("val_loss")
        val_acc = trainer.callback_metrics.get("val_acc")
        if val_loss is not None:
            self.val_loss.append(val_loss.item())
        if val_acc is not None:
            self.val_acc.append(val_acc.item())

    def get_metrics(self):
        return {
            "train_loss": self.train_loss,
            "val_loss": self.val_loss,
            "train_acc": self.train_acc,
            "val_acc": self.val_acc,
        }
