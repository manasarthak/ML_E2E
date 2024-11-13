import sys
from dataclasses import dataclass
import os
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifact', 'preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        try:
            # Columns to scale and one-hot encode
            scale_cols = ['person_income', 'person_age', 'person_emp_length', 'loan_amnt', 'loan_int_rate', 'cb_person_cred_hist_length', 'loan_percent_income']
            ohe_cols = ['cb_person_default_on_file', 'loan_grade', 'person_home_ownership', 'loan_intent']

            # Scaling pipeline
            scale_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ])

            # One-hot encoding pipeline
            ohe_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('onehot', OneHotEncoder(handle_unknown='ignore'))
            ])

            # Combine transformations
            preprocessor = ColumnTransformer(
                transformers=[
                    ('scale', scale_pipeline, scale_cols),
                    ('ohe', ohe_pipeline, ohe_cols)
                ]
            )
            logging.info('Data transformer object created successfully.')
            return preprocessor
        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        try:
            # Read train and test sets
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info('Loaded train and test data for transformation.')

            preprocessor = self.get_data_transformer_object()

            # Transform features
            X_train = train_df.drop(['loan_status'], axis=1)
            y_train = train_df['loan_status']
            X_test = test_df.drop(['loan_status'], axis=1)
            y_test = test_df['loan_status']

            X_train_transformed = preprocessor.fit_transform(X_train)
            X_test_transformed = preprocessor.transform(X_test)

            # Save the preprocessor object
            save_object(self.data_transformation_config.preprocessor_obj_file_path, preprocessor)
            logging.info('Preprocessor saved and transformation completed.')

            return X_train_transformed, X_test_transformed, y_train, y_test,preprocessor
        except Exception as e:
            raise CustomException(e, sys)
