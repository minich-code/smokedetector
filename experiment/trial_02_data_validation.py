
from dataclasses import dataclass
from pathlib import Path
import json
import pandas as pd
import requests  # For sending Slack notifications
from scipy.stats import entropy, zscore

from src.smthome.constants import *
from src.smthome.utils.commons import read_yaml, create_directories
from src.smthome.logger import logger

@dataclass
class DataValidationConfig:
    root_dir: Path
    val_status: str
    data_dir: Path
    all_schema: dict
    critical_columns: list
    slack_webhook_url: str

class ConfigurationManager:
    def __init__(
        self, 
        data_validation_config: Path = DATA_VALIDATION_CONFIG_FILEPATH, 
        schema_config: Path = SCHEMA_CONFIG_FILEPATH
    ) -> None:
        logger.info("Initializing ConfigurationManager")
        self.data_val_config = read_yaml(data_validation_config)
        self.schema = read_yaml(schema_config)
        create_directories([self.data_val_config.artifacts_root])

    def get_data_validation_config(self) -> DataValidationConfig:
        data_valid_config = self.data_val_config.data_validation
        schema = self.schema.get('columns', {})
        try:
            create_directories([data_valid_config.root_dir])
        except CustomException as e:
            logger.error(f"Error creating directories: {str(e)}")
            raise
        
        logger.debug("Data validation configuration loaded")
        return DataValidationConfig(
            root_dir=data_valid_config.root_dir,
            val_status=data_valid_config.val_status,
            data_dir=data_valid_config.data_dir,
            all_schema=schema,
            critical_columns=data_valid_config.critical_columns,
            slack_webhook_url=data_valid_config.slack_webhook_url
        )

class DataValidation:
    def __init__(self, config: DataValidationConfig):
        self.config = config
        logger.info("DataValidation initialized with config")

    def _send_slack_notification(self, message: str):
        """Sends a notification to Slack."""
        payload = {'text': message}
        try:
            response = requests.post(self.config.slack_webhook_url, json=payload)
            response.raise_for_status()
            logger.info("Slack notification sent successfully")
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to send Slack notification: {e}")

    def _validate_columns(self, data):
        """Validates if all expected columns are present and no unexpected columns exist."""
        all_cols = list(data.columns)
        all_schema = list(self.config.all_schema.keys())

        missing_columns = [col for col in all_schema if col not in all_cols]
        extra_columns = [col for col in all_cols if col not in all_schema]

        error_message = {"missing_columns": missing_columns, "extra_columns": extra_columns}
        if missing_columns or extra_columns:
            logger.debug(f"Validation failed for columns: {error_message}")
            return False, error_message
        return True, None

    def _validate_data_types(self, data):
        """Validates if the data types of each column match the expected schema."""
        type_mapping = {
            "string": "object",
            "integer": "int64",
            "float": "float64",
        }

        all_schema = self.config.all_schema
        type_mismatches = {}
        validation_status = True

        for col, expected_type in all_schema.items():
            if col in data.columns:
                actual_type = str(data[col].dtype)
                if isinstance(expected_type, dict):
                    expected_type = expected_type.get("type", None)
                if not isinstance(expected_type, str):
                    logger.debug(f"Unexpected type format for column '{col}': {expected_type}")
                    expected_type = None
                expected_pandas_type = type_mapping.get(expected_type, None)
                if expected_pandas_type and actual_type != expected_pandas_type:
                    type_mismatches[col] = [expected_type, actual_type]
                    validation_status = False
                    
        if type_mismatches:
            logger.debug(f"Type mismatches: {type_mismatches}")
        return validation_status, type_mismatches

    def _validate_missing_values(self, data):
        """Validates if critical columns have any missing values."""
        missing_values = {}
        for col in self.config.critical_columns:
            if col not in data.columns:
                missing_values[col] = "Column not found"
            elif data[col].isnull().sum() > 0:
                missing_values[col] = data[col].isnull().sum()
                
        if missing_values:
            logger.debug(f"Missing values: {missing_values}")
            return False
        return True

    def _check_cardinality(self, data):
        """Checks and drops columns with unique values."""
        drop_columns = [col for col in data.columns if data[col].nunique() == len(data)]
        if drop_columns:  # Only print if there are columns to drop
            logger.info(f"Dropping columns with unique values: {drop_columns}")
        data.drop(columns=drop_columns, inplace=True)
        logger.debug(f"Dropped columns with unique values: {drop_columns}")
        return data

    def validate_data(self, data):
        """Performs all data validation checks and returns the overall validation status."""
        validation_results = {}
        
        # Validate all columns
        status, error_message = self._validate_columns(data)
        validation_results["validate_all_columns"] = {"status": status, "error_message": error_message}
        
        # Validate data types
        type_validation_status, type_mismatches = self._validate_data_types(data)
        validation_results["validate_data_types"] = {
            "status": type_validation_status,
            "mismatches": type_mismatches
        }
        
        # Validate missing values
        validation_results["validate_missing_values"] = {"status": self._validate_missing_values(data)}
        
        # Save results to file
        with open(self.config.val_status, 'w') as f:
            json.dump(validation_results, f, indent=4)
        
        overall_validation_status = all(result["status"] for result in validation_results.values())

        # Send Slack notification
        if overall_validation_status:
            message = "Data validation completed successfully."
            logger.info(message)
            self._send_slack_notification(message)
        else:
            message = "Data validation failed. Check the status file for details."
            logger.warning(message)
            self._send_slack_notification(message)
        
        # Save the validated data to a parquet file if validation is successful
        if overall_validation_status:
            output_path = str(Path(self.config.root_dir) / 'smoke_iot_data.parquet')
            data.to_parquet(output_path, index=False)
            logger.info(f"Validated data saved to: {output_path}")
        
        return overall_validation_status

if __name__ == "__main__":
    try:
        config_manager = ConfigurationManager()
        data_validation_config = config_manager.get_data_validation_config()
        data_validation = DataValidation(config=data_validation_config)
        data = pd.read_parquet(data_validation_config.data_dir)

        logger.info("Starting data validation process")
        validation_status = data_validation.validate_data(data)

        if validation_status:
            logger.info("Data Validation Completed Successfully!")
        else:
            logger.warning("Data Validation Failed. Check the status file for more details.")

    except Exception as e:
        logger.error(f"Data validation process failed: {e}")
        raise
