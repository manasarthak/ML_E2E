import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
import os

from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifact','preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()

    def get_data_transformer_object(self):
        '''
        Responsible for data transformation on columns of interest
        '''
        try:
            columns_of_interest=['TransactionAmount', 'CustomerAge']

            coi_pipeline=Pipeline(
                steps=[
                    ('imputer',SimpleImputer(strategy='median')),#not needed in our data as we dont have missing value,just for understanding pipeline
                    ('scaler',StandardScaler())
                ]
            )
            
            logging.info('Standard Scaling done for columns of interest')

            preprocessor=ColumnTransformer(
                [
                    ("coi_pipeline",coi_pipeline,columns_of_interest)
                ]
            )

            return preprocessor
        except Exception as e:
            raise CustomException(e,sys)
        
    def initiate_data_transformation(self,path):
        try:
            df=pd.read_csv(path)
            logging.info('Raw Data Read Completed')
            logging.info('Obtaining Preprocessing Object')
            preprocessor_obj=self.get_data_transformer_object()

            columns_of_interest=['TransactionAmount', 'CustomerAge']
            final_df=df[columns_of_interest]

            logging.info('Applying preprocessing object on subset of dataframe i.e usefl for anomaly detection')

            final_scaled=preprocessor_obj.fit_transform(final_df)
            
            logging.info('Saved Preprocessing object')

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessor_obj
            )

            return(
                final_scaled,
                self.data_transformation_config.preprocessor_obj_file_path
            )

        except Exception as e:
            raise CustomException(e,sys)


