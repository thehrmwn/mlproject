# Read dataset from specific sources (read, split)

import os
import sys
from src.exception import CustomException
from src.logger import logging

import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.components.data_transformation import DataTransformationConfig
from src.components.data_transformation import DataTransformation

from src.components.model_trainer import ModelTrainerConfig
from src.components.model_trainer import ModelTrainer


@dataclass
class DataIngestionConfig:
    # @dataclass when a class but doesn't have a function
    train_data_path:str = os.path.join('artifacts', 'train.csv') 
    test_data_path:str = os.path.join('artifacts', 'test.csv') 
    raw_data_path:str = os.path.join('artifacts', 'data.csv') 
    
class DataIngestion:
    def __init__(self):
        # create variable to save the 3 variables from DataIngestionConfig
        self.ingestion_config = DataIngestionConfig()
        
    def initiate_data_ingestion(self):
        logging.info("Entered the Data Ingestion method or components")
        
        try:
            # change the code to read from another resource (MongoDB, API, databases, etc.)
            df = pd.read_csv("notebook\data\student.csv")
            logging.info("Read the dataset")
            
            # make directory for train-test data (artifact)
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)
            
            # data to raw file
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)
            
            # Train test split
            logging.info("Train Test split initiated")
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)
            logging.info("Ingestion of the Data is Completed")
            
            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
            
        except Exception as e:
            raise CustomException(e, sys)


        
# Test the ingestion
if __name__ == "__main__":
    obj = DataIngestion()
    train_data, test_data = obj.initiate_data_ingestion()
    
    data_transformation = DataTransformation()
    train_arr, test_arr, _ = data_transformation.iniatiate_data_tranformation(train_data, test_data)
    
    model_trainer = ModelTrainer()
    best_model = model_trainer.initiate_model_trainer(train_arr, test_arr)
    print(f"Best Model: {best_model[0]} \n \Score: {best_model[1]}")