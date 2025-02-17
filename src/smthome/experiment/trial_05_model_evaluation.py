
from pathlib import Path
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (classification_report, confusion_matrix, roc_curve, auc, 
                             precision_recall_curve, f1_score, precision_score, 
                             recall_score, roc_auc_score, accuracy_score)
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import itertools
from src.smthome.constants import *
from src.smthome.utils.commons import save_json, read_yaml, create_directories
from src.smthome.logger import logger
from src.smthome.exception import CustomException
from src.smthome.model.model import FireAlarmClassifier
import json
from src.smthome.model.model import FireAlarmClassifier

@dataclass
class ModelEvaluationConfig:
    root_dir: Path
    val_features_path: Path
    val_target_path: Path
    model_path: Path
    metric_file_name: Path
    validation_metrics_path: Path
    model_name: Path
    training_metrics_path: Path
    report_path: Path
    confusion_matrix_report: Path


class ConfigurationManager:
    def __init__(self, model_evaluation_config: str = MODEL_EVALUATION_CONFIG_FILEPATH, params_config: str = PARAMS_CONFIG_FILEPATH):
        self.model_evaluation_config = read_yaml(model_evaluation_config)
        self.params = read_yaml(params_config)
        create_directories([Path(self.model_evaluation_config['artifacts_root'])])

    def get_model_evaluation_config(self) -> ModelEvaluationConfig:
        eval_config = self.model_evaluation_config['model_evaluation']
        params = self.params['dnn_params']
        create_directories([Path(eval_config['root_dir'])])
        logger.info("Getting model evaluation config")
        return ModelEvaluationConfig(
            root_dir=Path(eval_config['root_dir']),
            val_features_path=Path(eval_config['val_features_path']),
            val_target_path=Path(eval_config['val_target_path']),
            model_path=Path(eval_config['model_path']),
            metric_file_name=Path(eval_config['metric_file_name']),
            validation_metrics_path=Path(eval_config['validation_metrics_path']),
            model_name=Path(eval_config['model_name']),
            training_metrics_path=Path(eval_config['training_metrics_path']),
            report_path=Path(eval_config['report_path']),
            confusion_matrix_report=Path(eval_config['confusion_matrix_report']),
        )


class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig):
        self.config = config
        self.model = None
        self.val_loader = None
        self.y_true = []
        self.y_pred = []
        self.y_scores = []

    def load_model(self):
        logger.info(f"Loading model from {self.config.model_path}")
        checkpoint = torch.load(self.config.model_path, map_location=torch.device('cpu'))
        input_dim = checkpoint['input_dim']  # Retrieve 'input_dim' from checkpoint

        self.model = FireAlarmClassifier(input_dim=input_dim, output_dim=1, learning_rate=0.001)  # Initialize with input_dim
        self.model.load_state_dict(checkpoint['state_dict'])
        self.model.eval()
        logger.info("Model loaded successfully.")

    def load_data(self):
        logger.info("Loading test data.")
        X_val = torch.load(self.config.val_features_path)
        y_val = torch.load(self.config.val_target_path)
        dataset = TensorDataset(X_val, y_val)
        self.val_loader = DataLoader(dataset, batch_size=128, shuffle=False, num_workers=3, persistent_workers=True)
        logger.info("Validation data loaded successfully.")

    def evaluate(self):
        logger.info("Starting model evaluation on validation data.")
        self.model.eval()
        with torch.no_grad():
            for batch in self.val_loader:
                x, y = batch
                logits = self.model(x)
                probs = torch.sigmoid(logits)
                preds = (probs > 0.5).float()
                self.y_true.extend(y.cpu().numpy())
                self.y_pred.extend(preds.cpu().numpy())
                self.y_scores.extend(probs.cpu().numpy())
        logger.info("Model evaluation completed.")


    def generate_classification_report(self):
        logger.info("Generating classification report.")
        report = classification_report(self.y_true, self.y_pred, output_dict=True)
        report_path = os.path.join(self.config.report_path, "classification_report.txt") # self.config.report_path / "classification_report.txt"
        
        with open(self.config.report_path, "w") as f:
            f.write(classification_report(self.y_true, self.y_pred))
        
        logger.info(f"Classification report saved to {report_path}")

    def generate_confusion_matrix(self):
        logger.info("Generating confusion matrix.")
        cm = confusion_matrix(self.y_true, self.y_pred)
        cm_path = self.config.confusion_matrix_report
        plt.figure(figsize=(6, 6))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Confusion Matrix (Validation Data)')
        plt.colorbar()
        tick_marks = np.arange(2)  # Assuming two classes (0 and 1)
        plt.xticks(tick_marks, ['No Fire', 'Fire'], rotation=45)
        plt.yticks(tick_marks, ['No Fire', 'Fire'])
        fmt = 'd'
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.savefig(os.path.join(self.config.root_dir, "confusion_matrix.png"))
        plt.close()


    def generate_precision_recall_curve(self):
        logger.info("Generating Precision-Recall curve.")
        precision, recall, _ = precision_recall_curve(self.y_true, self.y_scores)
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, label='Precision-Recall Curve (Validation Data)')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve (Validation Data)')
        plt.legend()
        plt.savefig(os.path.join(self.config.root_dir, "precision_recall_curve.png"))
        plt.close()


    def generate_roc_curve(self):
        logger.info("Generating ROC curve.")
        fpr, tpr, thresholds = roc_curve(self.y_true, self.y_scores)
        roc_auc = auc(fpr, tpr)
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.2f}) (Validation Data)')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve (Validation Data)')
        plt.legend()
        plt.savefig(os.path.join(self.config.root_dir, 'roc_curve.png'))  
        plt.close()


    def save_metrics(self):
        logger.info("Saving evaluation metrics.")
        metrics = {
            "accuracy": accuracy_score(self.y_true, self.y_pred),
            "roc_auc": roc_auc_score(self.y_true, self.y_scores),
            "precision": precision_score(self.y_true, self.y_pred),
            "recall": recall_score(self.y_true, self.y_pred),
            "f1_score": f1_score(self.y_true, self.y_pred, average='weighted'),  # Use weighted average
        }
        metric_file = Path(self.config.root_dir) / "evaluation_metrics.json"
        with open(metric_file, "w") as f:
            json.dump(metrics, f, indent=4) 
        logger.info(f"Metrics saved to {metric_file}")

    def run_evaluation(self):
        self.load_model()
        self.load_data()
        self.evaluate()
        self.generate_classification_report()
        self.generate_confusion_matrix()
        self.generate_precision_recall_curve()
        self.generate_roc_curve()
        self.save_metrics()


if __name__ == "__main__":
    try:
        # Initialize Configuration Manager
        config_manager = ConfigurationManager()
        model_eval_config = config_manager.get_model_evaluation_config()

        # Initialize Model Evaluation
        model_evaluation = ModelEvaluation(config=model_eval_config)

        # Run Evaluation
        model_evaluation.run_evaluation()

        logger.info("Model evaluation completed successfully.")

    except Exception as e:
        logger.error(f"Error during model evaluation: {e}")
        raise CustomException(e, sys)
