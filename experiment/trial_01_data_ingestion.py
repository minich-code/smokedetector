
# File: data_ingestion.py

from dataclasses import dataclass
from pathlib import Path
import pymongo
import pandas as pd
import numpy as np
import os
import json
import time
import requests # for slack notifications

from datetime import datetime
from src.smthome.exception import CustomException
from src.smthome.utils.commons import *
from src.smthome.constants import *
from src.smthome.logger import logger


@dataclass
class DataIngestionConfig:
    root_dir: Path
    mongo_uri: str
    database_name: str
    collection_name: str
    batch_size: int
    slack_webhook_url: str

class ConfigurationManager:
    def __init__(self, data_ingestion_config: str = DATA_INGESTION_CONFIG_FILEPATH):
        self.ingestion_config = read_yaml(data_ingestion_config)
        create_directories([self.ingestion_config.artifacts_root])

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        data_config = self.ingestion_config.data_ingestion
        create_directories([data_config.root_dir])

        return DataIngestionConfig(**data_config)

class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config
        self.client = pymongo.MongoClient(self.config.mongo_uri)
        self.db = self.client[self.config.database_name]
        self.collection = self.db[self.config.collection_name]

    def send_slack_notification(self, message):
        payload = {"text": message}
        try:
            response = requests.post(self.config.slack_webhook_url, json=payload)
            response.raise_for_status()
            logger.info("Slack notification sent successfully")
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to send notification to Slack: {e}")

    def import_data_from_mongodb(self):
        start_time = time.time()
        start_timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        try:
            logger.info(f"Starting data ingestion from MongoDB collection {self.config.collection_name}")
            all_data = self.fetch_all_data()

            # Save the data to a Parquet file
            output_path = self.save_data(all_data)
            total_records = len(all_data)
            logger.info(f"Total records ingested: {total_records}")
            self._save_metadata(start_timestamp, start_time, total_records, output_path)

            # Send Slack notification
            self.send_slack_notification(f"Data ingestion completed successfully. Total records: {total_records}. Output path: {output_path}")

        except Exception as e:
            logger.error(f"Error during data ingestion: {e}")
            self.send_slack_notification(f"Data ingestion failed: {e}")
            raise CustomException(f"Error during data ingestion: {e}")

    def fetch_all_data(self):
        try:
            data_cursor = self.collection.find({}, {'_id': 0})
            data = list(data_cursor)
            df = pd.DataFrame(data)
            # Replace infinite values with NaN and drop rows with NaN values
            df.replace([np.inf, -np.inf], np.nan, inplace=True)
            df = df.dropna()

            logger.info(f"Data fetched successfully from MongoDB.")
            return df
        except Exception as e:
            logger.error(f"Error fetching data from MongoDB: {e}")
            raise CustomException(f"Error fetching data from MongoDB: {e}")

    def save_data(self, all_data):
        output_path = str(Path(self.config.root_dir) / 'smoke_data.parquet')
        all_data.to_parquet(output_path, index=False)
        logger.info(f"Data fetched from MongoDB and saved to {output_path}")
        return output_path

    def _save_metadata(self, start_timestamp, start_time, total_records, output_path):
        try:
            end_time = time.time()
            end_timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            duration = end_time - start_time
            metadata = {
                'start_time': start_timestamp,
                'end_time': end_timestamp,
                'duration': duration,
                'total_records': total_records,
                'data_source': self.config.collection_name,
                'output_path': output_path
            }
            metadata_path = Path(self.config.root_dir) / 'data-ingestion-metadata.json'
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=4)
            logger.info(f"Metadata saved to {metadata_path}")
        except Exception as e:
            logger.error(f"Error during JSON serialization: {e}")
            raise CustomException(f"Error saving metadata: {e}")

if __name__ == "__main__":
    try:
        config_manager = ConfigurationManager()
        data_ingestion_config = config_manager.get_data_ingestion_config()
        data_ingestion = DataIngestion(config=data_ingestion_config)
        data_ingestion.import_data_from_mongodb()
        logger.info("Data Ingestion from MongoDB Completed!")
    except CustomException as e:
        logger.error(f"Data ingestion process failed: {e}")






