import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd

from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig

from src.components.model_trainer import ModelTrainer
from src.components.model_trainer import ModelTrainerConfig

@dataclass
class DataIngestionConfig:
    train_data_path: str= os.path.join('artifact','train_data.csv')
    test_data_path: str= os.path.join('artifact','test_data.csv')
    raw_data_path: str= os.path.join('artifact','raw_data.csv')

class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionConfig()
    
    def initiate_data_ingestion(self):
        logging.info('Entered the data ingestion method')
        try:
            #simple read for now
            df=pd.read_csv('notebooks_for_EDA/dataset/credit_risk_dataset.csv')
            logging.info('Dataset read')

            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path),exist_ok=True)

            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)

            logging.info('Train-test split initiated')
            train_set,test_set=train_test_split(df,test_size=0.2,random_state=42)
            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)
            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)
            logging.info('Ingestion Complete')

            return(
                self.ingestion_config.raw_data_path,
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path,
            )
        except Exception as e:
            raise CustomException(e,sys)
        
if __name__ == "__main__":
    # Step 1: Data Ingestion
    obj = DataIngestion()
    data_path,train_data_path,test_data_path = obj.initiate_data_ingestion()

    # Step 2: Data Transformation
    data_transformation = DataTransformation()
    X_train_transformed,X_test_transformed,y_train,y_test, preprocessor_path = data_transformation.initiate_data_transformation(train_data_path,test_data_path)

    # Step 3: Model Training
    model_trainer = ModelTrainer()
    model_report = model_trainer.initiate_model_trainer(X_train_transformed, X_test_transformed, y_train, y_test)
    
    print("Model Report:", model_report)