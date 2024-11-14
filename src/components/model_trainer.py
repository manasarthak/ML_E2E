import os
import sys
from dataclasses import dataclass

from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from sklearn.model_selection import RandomizedSearchCV

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts', 'model')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def train_classifier(self, clf, X_train, y_train, X_test, y_test):
        """Train the classifier and calculate evaluation metrics"""
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)

        # Calculate confusion matrix to get TN and FP for specificity
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        specificity = tn / (tn + fp)

        return accuracy, precision, recall, specificity

    def initiate_model_trainer(self, X_train, X_test, y_train, y_test):
        try:
            logging.info('Initializing model training for classification')

            # Define classification models
            models = {
                'knn': KNeighborsClassifier(),
                'xgb': XGBClassifier(),
                'cat': CatBoostClassifier(verbose=0),
                'lgb': LGBMClassifier()
            }

            # Hyperparameter grids for RandomizedSearchCV
            param_grids = {
                'knn': {
                    'n_neighbors': [3, 5, 7, 9],
                    'weights': ['uniform', 'distance'],
                    'p': [1, 2]
                },
                'xgb': {
                    'n_estimators': [50, 100, 200],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'max_depth': [3, 5, 7]
                },
                'cat': {
                    'iterations': [100, 200, 300],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'depth': [3, 6, 10]
                },
                'lgb': {
                    'num_leaves': [15, 31, 63],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'max_depth': [3, 5, 7]
                }
            }

            # Perform hyperparameter tuning for each model
            tuned_models = {}
            for model_name, model in models.items():
                logging.info(f"Tuning hyperparameters for {model_name}")
                random_search = RandomizedSearchCV(
                    model,
                    param_distributions=param_grids[model_name],
                    n_iter=10,
                    scoring='recall',
                    cv=5,
                    random_state=42,
                    n_jobs=-1
                )
                random_search.fit(X_train, y_train)
                tuned_models[model_name] = random_search.best_estimator_
                logging.info(f"Best parameters for {model_name}: {random_search.best_params_}")

            # Ensemble Voting Classifier (Soft Voting)
            ensemble = VotingClassifier(
                estimators=[('knn', tuned_models['knn']), 
                            ('xgb', tuned_models['xgb']),
                            ('cat', tuned_models['cat']),
                            ('lgb', tuned_models['lgb'])],
                voting='soft'
            )
            logging.info("Training ensemble model with soft voting.")
            ensemble.fit(X_train, y_train)

            # Evaluate models
            accuracy_scores = {}
            precision_scores = {}
            recall_scores = {}
            specificity_scores = {}

            for model_name, model in {**tuned_models, 'ensemble': ensemble}.items():
                accuracy, precision, recall, specificity = self.train_classifier(model, X_train, y_train, X_test, y_test)
                accuracy_scores[model_name] = accuracy
                precision_scores[model_name] = precision
                recall_scores[model_name] = recall
                specificity_scores[model_name] = specificity

                logging.info(f"{model_name} - Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, Specificity: {specificity}")

            # Save each tuned model individually
            os.makedirs(self.model_trainer_config.trained_model_file_path, exist_ok=True)
            for model_name, model in {**tuned_models, 'ensemble': ensemble}.items():
                model_file_path = os.path.join(self.model_trainer_config.trained_model_file_path, f"{model_name}.pkl")
                save_object(model_file_path, model)
                logging.info(f"Model '{model_name}' saved to {model_file_path}")

            # Return the evaluation metrics for further analysis
            model_report = {
                "accuracy": accuracy_scores,
                "precision": precision_scores,
                "recall": recall_scores,
                "specificity": specificity_scores
            }

            return model_report

        except Exception as e:
            raise CustomException(e, sys)
