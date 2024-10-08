
import setuptools
import logging
from dataclasses import dataclass
from pathlib import Path
import os
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
from src.smthome.logger import logger
from src.smthome.exception import CustomException
from src.smthome.utils.commons import save_json, read_yaml, create_directories
from src.smthome.constants import *


@dataclass()
class ModelTrainerConfig:
    root_dir: Path
    train_features_path: Path
    train_targets_path: Path
    val_features_path: Path
    val_targets_path: Path
    val_metrics_path: Path
    model_name: str
    batch_size: int
    learning_rate: float


class ConfigurationManager:
    def __init__(
        self,
        model_training_config=MODEL_TRAINER_CONFIG_FILEPATH,
        params_config=PARAMS_CONFIG_FILEPATH,
    ):
        self.training_config = read_yaml(model_training_config)
        self.params = read_yaml(params_config)
        create_directories([self.training_config.artifacts_root])

    def get_model_trainer_config(self) -> ModelTrainerConfig:
        trainer_config = self.training_config.model_trainer
        params = self.params.dnn_params

        create_directories([trainer_config.root_dir])

        return ModelTrainerConfig(
            root_dir=Path(trainer_config.root_dir),
            train_features_path=trainer_config.train_features_path,
            train_targets_path=trainer_config.train_targets_path,
            val_features_path=trainer_config.val_features_path,
            val_targets_path=trainer_config.val_targets_path,
            val_metrics_path=trainer_config.val_metrics_path,
            model_name=trainer_config.model_name,
            batch_size=params.batch_size,
            learning_rate=params.learning_rate,
        )


class FireAlarmClassifier(pl.LightningModule):
    def __init__(self, input_dim, output_dim, learning_rate):
        super(FireAlarmClassifier, self).__init__()
        self.layer_1 = nn.Linear(input_dim, 16)
        self.layer_2 = nn.Linear(16, 8)
        self.layer_3 = nn.Linear(8, 4)
        self.layer_4 = nn.Linear(4, 1)
        self.loss_fn = nn.BCEWithLogitsLoss()
        self.learning_rate = learning_rate

    def forward(self, x):
        x = F.relu(self.layer_1(x))
        x = F.relu(self.layer_2(x))
        x = F.relu(self.layer_3(x))
        x = self.layer_4(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        y = y.unsqueeze(1).float()
        loss = self.loss_fn(logits, y)

        # Calculate accuracy
        preds = (torch.sigmoid(logits) > 0.5).float()  # Convert logits to predictions
        acc = (preds == y).float().mean()
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train_acc", acc, on_step=False, on_epoch=True, prog_bar=True) # log accuracy
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        y = y.unsqueeze(1).float()
        val_loss = self.loss_fn(logits, y)

        # Calculate accuracy
        preds = (torch.sigmoid(logits) > 0.5).float()  # Convert logits to predictions
        acc = (preds == y).float().mean()
        self.log("val_loss", val_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_acc", acc, on_step=False, on_epoch=True, prog_bar=True) # log accuracy

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        y = y.unsqueeze(1).float()
        test_loss = self.loss_fn(logits, y)
        
        # Calculate accuracy
        preds = (torch.sigmoid(logits) > 0.5).float()
        acc = (preds == y).float().mean()
        self.log("test_loss", test_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test_acc", acc, on_step=False, on_epoch=True, prog_bar=True)
      
    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.learning_rate)
        return optimizer


class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config

    def train(self):
        try:
            # Load data directly inside the train method
            train_features = torch.load(self.config.train_features_path)
            train_targets = torch.load(self.config.train_targets_path)
            val_features = torch.load(self.config.val_features_path)
            val_targets = torch.load(self.config.val_targets_path)

            train_dataset = TensorDataset(train_features, train_targets)
            val_dataset = TensorDataset(val_features, val_targets)

            train_loader = DataLoader(
                train_dataset,
                batch_size=self.config.batch_size,
                shuffle=True,
                num_workers=3,
                persistent_workers=True,
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.config.batch_size,
                shuffle=False,
                num_workers=3,
                persistent_workers=True,
            )

            # Initialize the model
            input_dim = train_features.shape[1]
            model = FireAlarmClassifier(
                input_dim=input_dim,
                output_dim=1,
                learning_rate=self.config.learning_rate,
            )

            # Define the Trainer
            trainer = pl.Trainer(
                max_epochs=7, 
                callbacks=[MetricsCallback()], 
                accelerator='cpu'
            ) 

            # Train the model
            trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

            # Save the trained model (optional)
            torch.save(model.state_dict(), os.path.join(self.config.root_dir, 'fire_alarm_model.pth'))


            # Save the trained model's state_dict in .pt format
            model_save_path = os.path.join(self.config.root_dir, 'fire_alarm_model.pt')
            torch.save(model.state_dict(), model_save_path)
            logger.info(f"Model state_dict saved to {model_save_path}")

            # Save the checkpoint with additional information
            checkpoint = {
                'state_dict': model.state_dict(),
                'input_dim': input_dim,
                'output_dim': model.layer_4.out_features,  # Assuming output_dim is needed
                'learning_rate': model.learning_rate
            }
            torch.save(checkpoint, 'artifacts/model_trainer/fire_alarm_model.pt')

            # Plot metrics
            self.plot_metrics(trainer.callbacks[0])

        except Exception as e:
            logger.error(f"Error during training: {str(e)}")
            raise CustomException(f"Error during model training: {e}")

    def plot_metrics(self, metrics_callback):
        try:
            # Correcting lengths for consistent plotting
            min_epochs = min(
                len(metrics_callback.train_loss),
                len(metrics_callback.val_loss),
                len(metrics_callback.train_acc),
                len(metrics_callback.val_acc),
            )

            # Ensure all metric lists are of the same length
            epochs = range(1, min_epochs + 1)
            train_loss = metrics_callback.train_loss[:min_epochs]
            val_loss = metrics_callback.val_loss[:min_epochs]
            train_acc = metrics_callback.train_acc[:min_epochs]
            val_acc = metrics_callback.val_acc[:min_epochs]

            # Plot training and validation loss
            plt.figure(figsize=(10, 5))
            plt.plot(epochs, train_loss, label="Training Loss")
            plt.plot(epochs, val_loss, label="Validation Loss")
            plt.xlabel("Epochs")
            plt.ylabel("Loss")
            plt.title("Training and Validation Loss")
            plt.legend()
            plt.savefig(os.path.join(self.config.root_dir, 'loss_plot.png'))
            plt.show()

            # Plot training and validation accuracy
            plt.figure(figsize=(10, 5))
            plt.plot(epochs, train_acc, label="Training Accuracy")
            plt.plot(epochs, val_acc, label="Validation Accuracy")
            plt.xlabel("Epochs")
            plt.ylabel("Accuracy")
            plt.title("Training and Validation Accuracy")
            plt.legend()
            plt.savefig(os.path.join(self.config.root_dir, 'accuracy_plot.png'))
            plt.show()

        except Exception as e:
            logger.error(f"Error plotting metrics: {str(e)}")
            raise CustomException(f"Error during metrics plotting: {e}")


# Callback for metrics logging
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


if __name__ == '__main__':
    try:
        config_manager = ConfigurationManager()
        model_trainer_config = config_manager.get_model_trainer_config()
        model_trainer = ModelTrainer(config=model_trainer_config)
        model_trainer.train()

    except Exception as e:
        logger.exception(f"Error occurred: {e}")
        raise CustomException(e, sys)




