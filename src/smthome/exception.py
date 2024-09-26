import sys
import logging
from typing import Any

# Initialize logger
logger = logging.getLogger('SmokeDetector')

# Method to extract error message detail
def error_message_detail(error: Exception, error_details_object: Any) -> str:
    # Get the traceback information from error detail
    _, _, exc_tb = error_details_object.exc_info()

    # Extract the filename from the traceback
    file_name = exc_tb.tb_frame.f_code.co_filename

    # Create a formatted error message
    formatted_error_message = f"File: {file_name} \nLine Number: [{exc_tb.tb_lineno}] \nError message: [{error}]"
    
    return formatted_error_message

# Define the custom exception
class CustomException(Exception):
    def __init__(self, error: Exception, error_details_object: Any):
        # Extract the formatted error message
        formatted_error_message = error_message_detail(error, error_details_object)
        
        # Log the error message
        logger.error(formatted_error_message)
        
        # Call the base class constructor with the formatted error message
        super().__init__(formatted_error_message)
        
        self.error = error
        self.error_details_object = error_details_object

    # Override the __str__ method to return the formatted error message
    def __str__(self) -> str:
        return error_message_detail(self.error, self.error_details_object)