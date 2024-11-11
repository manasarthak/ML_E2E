import os
import sys

import numpy as np
import pandas as pd
import dill

from src.exception import CustomException
from src.logger import logging
from sklearn.metrics import silhouette_score

def save_object(file_path,obj):
    try:
        dir_path=os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)
        with open(file_path,'wb') as file_obj:#write byte mode
            dill.dump(obj,file_obj)
    
    except Exception as e:
        raise CustomException(e,sys)

from sklearn.metrics import silhouette_score
import numpy as np
import pandas as pd

def evaluate_model(data, models, required_flags=2):
    try:
        model_report = {}
        model_flags = {}

        for model_name, model in models.items():
            if model_name == 'kmeans':
                model.fit(data)
                labels = model.labels_
                silhouette_avg = silhouette_score(data, labels)
                distances = np.linalg.norm(data - model.cluster_centers_[labels], axis=1)
                threshold = np.percentile(distances, 95)
                fraud_flags = distances > threshold
                model_report[model_name] = {'silhouette_score': silhouette_avg, 'anomaly_count': fraud_flags.sum()}
                model_flags[model_name] = fraud_flags

            elif model_name == 'hierarchial':
                model.fit(data)
                labels = model.labels_
                silhouette_avg = silhouette_score(data, labels)
                cluster_distances = []
                for label in np.unique(labels):
                    cluster_points = data[labels == label]
                    if len(cluster_points) > 1:
                        intra_distances = np.linalg.norm(cluster_points - cluster_points.mean(axis=0), axis=1)
                        cluster_distances.extend(intra_distances)
                cluster_distances = np.array(cluster_distances)
                threshold = np.percentile(cluster_distances, 95)
                fraud_flags = cluster_distances > threshold
                model_report[model_name] = {'silhouette_score': silhouette_avg, 'anomaly_count': fraud_flags.sum()}
                model_flags[model_name] = fraud_flags

            elif model_name == 'DBSCAN':
                model.fit(data)
                outlier_count = (model.labels_ == -1).sum()
                model_report[model_name] = {'outlier_count': outlier_count}
                model_flags[model_name] = (model.labels_ == -1)

            elif model_name == 'iso_forest':
                model.fit(data)
                anomaly_scores = model.decision_function(data)
                threshold = np.percentile(anomaly_scores, 5)
                fraud_flags = anomaly_scores < threshold
                model_report[model_name] = {'anomaly_count': fraud_flags.sum()}
                model_flags[model_name] = fraud_flags

        # Combine flags from multiple models
        combined_flags = sum(flags.astype(int) for flags in model_flags.values()) >= required_flags
        model_report['combined'] = {'anomaly_count': combined_flags.sum()}

        return model_report, combined_flags
    
    except Exception as e:
        raise CustomException(e, sys)

