import pandas as pd
import torch
from pathlib import Path
from dataclasses import dataclass
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OrdinalEncoder, StandardScaler
from torch.utils.data import TensorDataset
from src.smthome.logger import logger

from src.smthome.exception import CustomException
from src.smthome.utils.commons import save_object, read_yaml, create_directories
from src.smthome.constants import *
import requests  # For Slack notification

import joblib

@dataclass 
class DataTransformationConfig:
    root_dir: Path 
    data_path: Path 
    slack_webhook_url: str

class ConfigurationManager:
    def __init__(self, data_preprocessing_config: str = DATA_TRANSFORMATION_FILEPATH):
        self.preprocessing_config = read_yaml(data_preprocessing_config)
        artifacts_root = self.preprocessing_config.artifacts_root
        create_directories([artifacts_root])

    def get_data_transformation_config(self) -> DataTransformationConfig:
        logger.info("Getting data transformation config")

        transformation_config = self.preprocessing_config.data_transformation
        create_directories([transformation_config.root_dir])

        return DataTransformationConfig(
            root_dir=Path(transformation_config.root_dir),
            data_path=Path(transformation_config.data_path),
            slack_webhook_url=transformation_config.slack_webhook_url
        )


class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config

    def load_data(self):
        logger.info("Reading data...")
        try:
            df = pd.read_parquet(self.config.data_path)
            logger.info(f"Data shape: {df.shape[0]} rows, {df.shape[1]} columns")
            return df
        except Exception as e:
            logger.error(f"Error reading data: {str(e)}")
            raise CustomException(e)

    def _send_slack_notification(self, message: str):
        """Sends a notification to Slack."""
        
        try:
            payload = {'text': message}
            response = requests.post(self.config.slack_webhook_url, json=payload)
            response.raise_for_status()
            logger.info("Slack notification sent successfully")
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to send Slack notification: {e}")

    def drop_unused_columns(self, df):
        logger.info("Dropping unused columns...")
        # Drop columns with unique values and zero variance 
        unique_val_columns = [col for col in df.columns if df[col].nunique() == 1]

        # Drop columns with zero variance only among numeric columns 
        zero_variance_columns = [col for col in df.select_dtypes(include='number').columns if df[col].var() == 0]


        columns_to_drop = unique_val_columns + zero_variance_columns + ['UTC']
        # Drop unwanted columns in place
        df.drop(columns=columns_to_drop, inplace=True, errors='ignore')
        logger.info(f"Data shape: {df.shape[0]} rows, {df.shape[1]} columns")
        
        return df

    def save_tensors(self, tensors, names):
        """Saves tensors to the specified directory."""
        logger.info("Saving tensors...")
        for tensor, name in zip(tensors, names):
            tensor_path = self.config.root_dir / f"{name}.pt"
            torch.save(tensor, tensor_path)
            logger.info(f"Saved tensor: {tensor_path}")
        logger.info("All tensors saved successfully.")

    def transformer_obj(self):
        logger.info("Creating transformer object...")
        transformer = MinMaxScaler()
        logger.info("Transformer object created.")
        return transformer

    
    def save_scaler(self, scaler):
        """Saves the scaler object for future use."""
        logger.info("Saving scaler object...")
        scaler_path = self.config.root_dir / "scaler.pkl"
        save_object(scaler, scaler_path)
        logger.info(f"Scaler saved at: {scaler_path}")

    def train_val_test_split(self):
        logger.info("Splitting data into train, validation, and test sets.")
        df = self.load_data()
        df = self.drop_unused_columns(df)

        logger.info(f"Data shape after dropping unused columns: {df.shape[0]} rows, {df.shape[1]} columns")



        # Drop Missing Values 
        df = df.dropna()

        # FEatures and Target variable 
        X = df.drop(columns=['Fire_Alarm'])
        y = df['Fire_Alarm']

        logger.info(f"Data shape after dropping target column: {X.shape[0]} rows, {X.shape[1]} columns")

        # Splitting into training and temporary sets (70% train and 30% temp)
        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

        # Splitting the temporary set into validation and test sets (15% val and 15% test)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)


        # Standardize the dataset using the transformer object 
        transformer = self.transformer_obj()

        X_train = transformer.fit_transform(X_train)
        X_val = transformer.transform(X_val)
        X_test = transformer.transform(X_test)

        transformer_path = self.config.root_dir / "transformer.joblib"
        save_object(transformer_path, transformer)


        # Convert to PyTorch tensors
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32)
        X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
        y_val_tensor = torch.tensor(y_val.values, dtype=torch.float32)
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
        y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32)

        # Create TensorDataset objects
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
        test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
        logger.info("TensorDatasets created.")

        # Save tensors
        tensors = [X_train_tensor, y_train_tensor, X_val_tensor, y_val_tensor, X_test_tensor, y_test_tensor]
        names = ['X_train', 'y_train', 'X_val', 'y_val', 'X_test', 'y_test']
        self.save_tensors(tensors, names)

        return train_dataset, val_dataset, test_dataset


if __name__ == "__main__":
    try:
        config_manager = ConfigurationManager()
        data_transformation_config = config_manager.get_data_transformation_config()
        data_transformation = DataTransformation(config=data_transformation_config)
        
        # Perform the transformation
        train_dataset, val_dataset, test_dataset = data_transformation.train_val_test_split()

    except CustomException as e:
        logger.error(f"Data transformation process failed: {e}")




        


