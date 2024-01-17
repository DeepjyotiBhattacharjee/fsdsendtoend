import pandas as pd
import numpy as np
from src.DimondPricePrediction.logger import logging
from sklearn.model_selection import train_test_split
from src.DimondPricePrediction.exception import customexception
from dataclasses import dataclass
from pathlib import Path
import os
import sys

class DataIngestionConfig:
    raw_data_path : str = os.path.join("artifacts","raw.csv")
    train_data_path : str = os.path.join("artifacts","train.csv")
    test_data_path : str =  os.path.join("artifacts","test.csv")


class DataIngestion:
    def __init__(self): 
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Data ingestion started.")

        try:
            print("current working dir === ",os.getcwd())
            data = pd.read_csv(Path(os.path.join("notebooks/data","gemstone.csv")))
            logging.info("Read the dataset as a dataframe")

            os.makedirs(os.path.join(self.ingestion_config.raw_data_path),exist_ok=True)
            data.to_csv(self.ingestion_config.raw_data_path,index=False)
            logging.info("Saved raw data path in artifacts folder.")

            logging.info("Performing train test split.") 
            train_data,test_data =  train_test_split(data ,test_size=0.25)
            train_data.to_csv(self.ingestion_config.train_data_path,index=False)
            test_data.to_csv(self.ingestion_config.test_data_path,index=False) 
            logging.info("Train test split completed")
            logging.info("Data ingestion completed.")
        except Exception as e:
            logging.info("Exception occured in data ingestion.")
            raise customexception(e,sys) 
