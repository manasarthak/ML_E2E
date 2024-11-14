import sys
import pandas as pd
from src.exception import CustomException
from src.logger import logging
from src.utils import load_object

class PredictPipeline:
    def __init__(self):
        # Paths to the saved models and preprocessor
        self.model_paths = {
            'knn': 'artifacts/model/knn.pkl',
            'xgb': 'artifacts/model/xgb.pkl',
            'cat': 'artifacts/model/cat.pkl',
            'lgb': 'artifacts/model/lgb.pkl',
            'ensemble': 'artifacts/model/ensemble.pkl'
        }
        self.preprocessor_path = 'artifacts/preprocessor.pkl'

    def predict(self, features):
        try:
            # Load preprocessor and scale the features
            preprocessor = load_object(file_path=self.preprocessor_path)
            data_scaled = preprocessor.transform(features)
            
            # Dictionary to store predictions for each model
            results = {}

            # Load and apply each model
            for model_name, model_path in self.model_paths.items():
                model = load_object(file_path=model_path)
                
                # Get prediction and default probability if available
                prediction = model.predict(data_scaled)
                prediction_proba = model.predict_proba(data_scaled)[:, 1] if hasattr(model, "predict_proba") else None

                results[model_name] = {
                    "predicted_class": int(prediction[0]),
                    "default_probability": float(prediction_proba[0]) if prediction_proba is not None else None
                }
            
            return results
        except Exception as e:
            raise CustomException(e, sys)

class CustomData:
    def __init__(self, **kwargs):
        self.data = kwargs

    def get_data_as_dataframe(self):
        """
        Convert the input data to a DataFrame.
        """
        return pd.DataFrame([self.data])
