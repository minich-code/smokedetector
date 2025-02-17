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
class ModelValidationConfig:
    root_dir: Path
    test_feature_path: Path
    test_target_path: Path
    model_path: Path
    model_name: str
    report_path: Path
    confusion_matrix_report: Path


class ConfigurationManager:
    def __init__(self, model_validation_config: str = MODEL_VALIDATION_CONFIG_FILEPATH):
        self.validation_config = read_yaml(model_validation_config)
        create_directories([self.validation_config.artifacts_root])

    def get_model_validation_config(self) -> ModelValidationConfig:
        val_config = self.validation_config.model_validation

        create_directories([val_config.root_dir])
        logger.info(f"Getting Model validation config: {val_config}")

        return ModelValidationConfig(
            root_dir=val_config.root_dir,
            test_feature_path=val_config.test_feature_path,
            test_target_path=val_config.test_target_path,
            model_path=val_config.model_path,
            model_name=val_config.model_name,
            report_path=val_config.report_path,
            confusion_matrix_report=val_config.confusion_matrix_report

        )


class ModelValidation:
    def __init__(self, config: ModelValidationConfig):
        self.config = config 
        self.model = None
        self.test_loader= None 
        self.y_true = []
        self.y_pred = []
        self.y_scores = []

    def load_model(self):
        logger.info(f"Loading model from {self.config.model_path}")
        checkpoint = torch.load(self.config.model_path, map_location=torch.device('cpu'))
        input_dim = checkpoint['input_dim']  # Retrieve 'input_dim' from checkpoint
        self.model = FireAlarmClassifier(input_dim=input_dim, output_dim=1, learning_rate=0.001)
        self.model.load_state_dict(checkpoint['state_dict'])
        self.model.eval()
        logger.info("Model loaded successfully.")  # Initialize with input_dim

    def load_test_data(self):
        logger.info(f"Loading test data from {self.config.test_feature_path}")
        test_features = torch.load(self.config.test_feature_path)
        test_target = torch.load(self.config.test_target_path)
        self.test_loader = DataLoader(TensorDataset(test_features, test_target), batch_size=128, shuffle=False, num_workers=3, persistent_workers=True)
        logger.info("Test data loaded successfully.")

    def validate(self):
        logger.info("Starting model validation on test data.")
        self.model.eval()
        with torch.no_grad():
            for batch in self.test_loader:
                x, y = batch
                logits = self.model(x)
                probs = torch.sigmoid(logits)
                preds = (probs > 0.5).float()
                self.y_true.extend(y.cpu().numpy())
                self.y_pred.extend(preds.cpu().numpy())
                self.y_scores.extend(probs.cpu().numpy())
        logger.info("Model validation completed.")

    # def generate_classification_report(self):
    #     logger.info("Generating classification report.")
    #     report = classification_report(self.y_true, self.y_pred, output_dict=True)
    #     report_path = self.config.report_path / "classification_report.txt"

    #     with open(self.config.report_path, "w") as f:
    #         f.write(classification_report(self.y_true, self.y_pred))

    #     logger.info(f"Classification report saved to {report_path}")

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
        plt.title('Confusion Matrix (Test Data)')
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
        logger.info("Generating precision-recall curve.")
        precision, recall, _ = precision_recall_curve(self.y_true, self.y_scores)
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, label='Precision-Recall Curve (Test Data)')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve (Test Data)')
        plt.legend()
        plt.savefig(os.path.join(self.config.root_dir, "precision_recall_curve.png"))
        plt.close()


    def generate_roc_curve(self):
        logger.info("Generating ROC curve.")
        fpr, tpr, thresholds = roc_curve(self.y_true, self.y_scores)
        roc_auc = auc(fpr, tpr)
        plt.figure(figsize=(10, 7))
        plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.2f}) (Testing Data)')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve (Testing Data)')
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
        metric_file = Path(self.config.root_dir) / "validation_metrics.json"
        with open(metric_file, "w") as f:
            json.dump(metrics, f, indent=4) 
        logger.info(f"Metrics saved to {metric_file}")

    def run_validator(self):
        self.load_model()
        self.load_test_data()
        self.validate()
        self.generate_classification_report()
        self.generate_confusion_matrix()
        self.generate_precision_recall_curve()
        self.generate_roc_curve()
        self.save_metrics()


if __name__ == "__main__":
    try:
        # Initialize Configuration Manager
        config_manager = ConfigurationManager()
        model_validation_config = config_manager.get_model_validation_config()

        # Initialize Model Validation
        model_validator = ModelValidation(config=model_validation_config)

        # Run Evaluation
        model_validator.run_validator()

        logger.info("Model validation completed successfully.")

    except Exception as e:
        logger.error(f"Error during model validation: {e}")
        raise CustomException(e, sys)


    
