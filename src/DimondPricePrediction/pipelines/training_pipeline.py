from src.DimondPricePrediction.components.data_ingestion import DataIngestion
from src.DimondPricePrediction.logger import logging
from src.DimondPricePrediction.exception import customexception

import os
import sys
import pandas as pd

obj = DataIngestion()

obj.initiate_data_ingestion()