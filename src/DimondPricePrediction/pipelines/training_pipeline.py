from src.DimondPricePrediction.components import data_transformation
from src.DimondPricePrediction.components import model_trainer
from src.DimondPricePrediction.components.data_ingestion import DataIngestion
from src.DimondPricePrediction.components.data_transformation import DataTransformation
from src.DimondPricePrediction.components.model_trainer import ModelTrainer
from src.DimondPricePrediction.logger import logging
from src.DimondPricePrediction.exception import customexception

import os
import sys
import pandas as pd

data_ingestion_obj = DataIngestion()

train_data_path,test_data_path = data_ingestion_obj.initiate_data_ingestion()

data_transformation_obj = DataTransformation()

train_arr,test_arr = data_transformation_obj.initiate_data_transformation(train_data_path,test_data_path)

model_trainer_obj = ModelTrainer()
model_trainer_obj.initate_model_training(train_array=train_arr,test_array=test_arr)

