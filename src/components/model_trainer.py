from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.ensemble import IsolationForest

import os
import sys

from dataclasses import dataclass

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object
from src.utils import evaluate_model
from src.utils import tune_hyperparameters

@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join('artifact','model')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()

    def initiate_model_trainer(self, final_scaled):
        try:
            logging.info('Initializing model training for anomaly detection')
            data = final_scaled

            # Define models with base parameters
            models = {
                'kmeans': KMeans(),
                'DBSCAN': DBSCAN(),
                'hierarchial': AgglomerativeClustering(),
                'iso_forest': IsolationForest()
            }

            # Define hyperparameter grids for each model
            param_grids = {
                'kmeans': {'n_clusters': [2, 3, 4, 5], 'n_init': [10, 20]},
                'DBSCAN': {'eps': [0.2, 0.3, 0.5], 'min_samples': [3, 5, 10]},
                'hierarchial': {'n_clusters': [2, 3, 4], 'linkage': ['ward', 'complete', 'average']},
                'iso_forest': {'n_estimators': [50, 100, 200], 'contamination': [0.05, 0.1]}
            }

            # Perform hyperparameter tuning
            tuned_models = tune_hyperparameters(models, param_grids, data)
            
            # Evaluate the tuned models and get combined flags
            model_report, combined_flags = evaluate_model(data=data, models=tuned_models)
            logging.info(f"Model evaluation completed. Report: {model_report}")

            # Save each tuned model individually
            os.makedirs(self.model_trainer_config.trained_model_file_path, exist_ok=True)
            for model_name, model in tuned_models.items():
                model_file_path = os.path.join(self.model_trainer_config.trained_model_file_path, f"{model_name}.pkl")
                save_object(model_file_path, model)
                logging.info(f"Tuned model '{model_name}' saved to {model_file_path}")

            # Return the report and combined flags for further analysis
            return model_report, combined_flags

        except Exception as e:
            raise CustomException(e, sys)
