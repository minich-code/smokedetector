import logging
import logging.config
import os
import sys
from datetime import datetime
from pythonjsonlogger import jsonlogger
from logging.handlers import RotatingFileHandler

# Define the logfile name using the current date and time
log_file_name = f"{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}.log"

# Create the path to the log file
log_file_path = os.path.join(os.getcwd(), "logs", log_file_name)

# Create the log directory if it doesn't exist
os.makedirs(os.path.dirname(log_file_path), exist_ok=True)

# Logger configuration
LOGGING_CONFIG = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'standard': {
            'format': '[%(asctime)s] %(name)s - %(levelname)s - %(message)s'
        },
        'detailed': {
            'format': '[%(asctime)s] %(name)s - %(levelname)s - %(module)s - %(funcName)s - %(lineno)d - %(message)s'
        },
        'json': {
            '()': 'pythonjsonlogger.jsonlogger.JsonFormatter',
            'format': '%(asctime)s %(name)s %(levelname)s %(module)s %(funcName)s %(lineno)d %(message)s'
        },
    },
    'handlers': {
        'file': {
            'level': 'DEBUG',  # Log all levels to the file
            'class': 'logging.handlers.RotatingFileHandler',
            'filename': log_file_path,
            'maxBytes': 10*1024*1024,  # 10 MB
            'backupCount': 5,
            'formatter': 'detailed',
        },
        'console': {
            'level': 'INFO',  # Log info and higher levels to the console
            'class': 'logging.StreamHandler',
            'stream': sys.stdout,
            'formatter': 'standard',
        },
    },
    'loggers': {
        'SmokeDetector': {
            'handlers': ['file', 'console'],
            'level': 'DEBUG',  # Set the minimum logging level to DEBUG
            'propagate': False
        }
    }
}

# Apply logging configuration
logging.config.dictConfig(LOGGING_CONFIG)

# Get the logger
logger = logging.getLogger('SmokeDetector')

