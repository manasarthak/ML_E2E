import os
import sys

import numpy as np
import pandas as pd
import dill

from src.exception import CustomException
from src.logger import logging
from sklearn.metrics import silhouette_score
from sklearn.model_selection import ParameterGrid

def save_object(file_path,obj):
    try:
        dir_path=os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)
        with open(file_path,'wb') as file_obj:#write byte mode
            dill.dump(obj,file_obj)
    
    except Exception as e:
        raise CustomException(e,sys)

def tune_hyperparameters(models, param_grids, data):
    try:
        tuned_models = {}
        
        for model_name, model in models.items():
            best_model = model
            best_score = -np.inf  # Initialize with a very low score
            best_params = None  # Track best parameters
            best_anomaly_count = None  # Track anomaly count for the best parameters
            
            # Get the parameter grid for the model
            param_grid = param_grids.get(model_name, {})
            
            # Iterate over all parameter combinations
            for params in ParameterGrid(param_grid):
                model.set_params(**params)  # Set model parameters
                model.fit(data)  # Train model
                
                # Evaluate model
                if model_name == 'kmeans':
                    # For KMeans, use centroid distances for anomaly detection
                    labels = model.labels_
                    score = silhouette_score(data, labels)
                    distances = np.linalg.norm(data - model.cluster_centers_[labels], axis=1)
                    threshold = np.percentile(distances, 95)
                    anomaly_count = np.sum(distances > threshold)
                
                elif model_name == 'hierarchial':
                    # For AgglomerativeClustering, calculate intra-cluster distances
                    labels = model.labels_
                    score = silhouette_score(data, labels)
                    
                    # Calculate intra-cluster distances
                    cluster_distances = []
                    for label in np.unique(labels):
                        cluster_points = data[labels == label]
                        cluster_mean = cluster_points.mean(axis=0)
                        intra_distances = np.linalg.norm(cluster_points - cluster_mean, axis=1)
                        cluster_distances.extend(intra_distances)
                    
                    cluster_distances = np.array(cluster_distances)
                    threshold = np.percentile(cluster_distances, 95)
                    anomaly_count = np.sum(cluster_distances > threshold)
                
                elif model_name == 'DBSCAN':
                    # For DBSCAN, count points labeled as noise
                    score = -np.sum(model.labels_ == -1)  # Negate to maximize "best" score
                    anomaly_count = np.sum(model.labels_ == -1)
                
                elif model_name == 'iso_forest':
                    # For Isolation Forest, use decision_function for anomaly scoring
                    score = model.decision_function(data).mean()
                    anomaly_scores = model.decision_function(data)
                    threshold = np.percentile(anomaly_scores, 5)
                    anomaly_count = np.sum(anomaly_scores < threshold)
                
                # Print or log the current parameter set and anomaly count
                print(f"Model: {model_name}, Params: {params}, Score: {score}, Anomaly Count: {anomaly_count}")
                
                # Update the best model if the current one has a higher score
                if score > best_score:
                    best_score = score
                    best_model = model
                    best_params = params
                    best_anomaly_count = anomaly_count
            
            # Store the best model for this algorithm
            tuned_models[model_name] = best_model
            print(f"Best Model for {model_name}: Params: {best_params}, Score: {best_score}, Anomaly Count: {best_anomaly_count}")
        
        return tuned_models
    except Exception as e:
        raise CustomException(e, sys)


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
    
def load_object(file_path):
    try:
        with open(file_path,'rb') as file_obj:
            return dill.load(file_obj)
    except Exception as e:
        raise CustomException(e,sys)


