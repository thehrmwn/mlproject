import logging
import os
from datetime import datetime

LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log" #time of log
logs_path = os.path.join(os.getcwd(), "logs", LOG_FILE) #path / folder of log
os.makedirs(logs_path, exist_ok=True) #if exist keep appending the logs

LOG_FILE_PATH = os.path.join(logs_path, LOG_FILE)

logging.basicConfig(
    filename = LOG_FILE_PATH,
    format = "[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level = logging.INFO
    
)

# Testing Logger
# if __name__ == "__main__":
#     logging.info("Logging Has Started")