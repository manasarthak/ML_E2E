import sys
import pandas as pd
from src.exception import CustomException
from src.logger import logging
from src.utils import load_object
import numpy as np

class PredictPipeline:
    def __init__(self):
        # Paths to the saved models and preprocessor
        self.model_paths = {
            'kmeans': 'artifact/model/kmeans.pkl',
            'dbscan': 'artifact/model/dbscan.pkl',
            'hierarchial': 'artifact/model/hierarchial.pkl',
            'iso_forest': 'artifact/model/iso_forest.pkl'
        }
        self.preprocessor_path = 'artifact/preprocessor.pkl'

    def predict(self, features):
        try:
            # Load preprocessor
            preprocessor = load_object(file_path=self.preprocessor_path)
            data_scaled = preprocessor.transform(features)
            
            # Load and apply each model
            results = {}
            for model_name, model_path in self.model_paths.items():
                model = load_object(file_path=model_path)
                
                if model_name == 'kmeans':
                    distances = model.transform(data_scaled).min(axis=1)
                    threshold = np.percentile(distances, 95)
                    anomaly_flag = distances > threshold
                    results[model_name] = {"anomaly": bool(anomaly_flag[0]), "distance": distances[0]}
                
                elif model_name == 'dbscan':
                    labels = model.fit_predict(data_scaled)
                    results[model_name] = {"anomaly": bool(labels[0] == -1), "label": labels[0]}
                
                elif model_name == 'hierarchial':
                    if len(data_scaled) < 2:
                        results[model_name] = {"anomaly": None, "error": "Insufficient samples for clustering"}
                    else:
                        labels = model.fit_predict(data_scaled)
                        distances = []
                        for label in np.unique(labels):
                            cluster_points = data_scaled[labels == label]
                            cluster_mean = cluster_points.mean(axis=0)
                            cluster_distances = np.linalg.norm(cluster_points - cluster_mean, axis=1)
                            distances.extend(cluster_distances)
                        threshold = np.percentile(distances, 95)
                        anomaly_flag = distances > threshold
                        results[model_name] = {"anomaly": bool(anomaly_flag[0]), "distance": distances[0]}
                
                elif model_name == 'iso_forest':
                    scores = model.decision_function(data_scaled)
                    threshold = np.percentile(scores, 5)
                    anomaly_flag = scores < threshold
                    results[model_name] = {"anomaly": bool(anomaly_flag[0]), "score": scores[0]}
            
            return results
        except Exception as e:
            raise CustomException(e, sys)


class CustomData:
    def __init__(self, transaction_amount: float, customer_age: int):
        self.transaction_amount = transaction_amount
        self.customer_age = customer_age

    def get_data_as_dataframe(self):
        # Convert input data to a DataFrame
        input_data = {
            'TransactionAmount': [self.transaction_amount],
            'CustomerAge': [self.customer_age]
        }
        return pd.DataFrame(input_data)
