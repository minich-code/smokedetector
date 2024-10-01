
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.optim import Adam
from src.smthome.model.callbacks import MetricsCallback 
from dataclasses import dataclass
from experiment.trial_04_model_trainer import ModelTrainer



class FireAlarmClassifier(nn.Module):
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