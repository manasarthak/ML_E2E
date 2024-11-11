from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.ensemble import IsolationForest

import os
import sys

from dataclasses import dataclass

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object
from src.utils import evaluate_model

@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join('artifacts','model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()

    def initiate_model_trainer(self,final_scaled):
        try:
            logging.info('importing data to be fed into models')
            #we dont need train and test splits, so simple initialization
            models = {
                'kmeans': KMeans(n_clusters=3),
                'DBSCAN': DBSCAN(eps=0.3, min_samples=5),
                'hierarchial': AgglomerativeClustering(n_clusters=3),
                'iso_forest': IsolationForest(contamination=0.05)
            }

            # Evaluate models and get combined flags
            model_report, combined_flags = evaluate_model(data=final_scaled, models=models)
            logging.info(f"Model evaluation completed. Report: {model_report}")

            # Ensure the directory exists before saving models
            os.makedirs(self.model_trainer_config.trained_model_file_path, exist_ok=True)

            # Save each model individually
            for model_name, model in models.items():
                model_file_path = os.path.join(self.model_trainer_config.trained_model_file_path, f"{model_name}.pkl")
                save_object(model_file_path, model)
                logging.info(f"Model '{model_name}' saved to {model_file_path}")


            # Return the report and combined flags for further analysis
            return model_report, combined_flags

        except Exception as e:
            raise CustomException(e, sys)




        except Exception as e:
            raise CustomException(e,sys)
