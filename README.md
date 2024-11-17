## Loan Default Prediction
### Overview
This repository contains the implementation of a Loan Default Prediction system, leveraging machine learning models to predict whether a loan applicant is likely to default based on various input features. The solution includes:
    Data Preprocessing and Feature Engineering
    Model Training, Evaluation, and Ensemble Learning
    Prediction Pipeline
    Web Application for User Interaction

### The application is deployed across three platforms:
    Docker (via Amazon ECS)
    AWS Elastic Beanstalk (without Docker)
    Microsoft Azure Web Apps

### Features
    User-friendly web interface for predicting loan defaults.
    Multiple deployment options for scalability and accessibility.
    Robust ML pipeline for preprocessing, inference, and model ensembling.
    Automated Continuous Integration (CI) and Continuous Delivery (CD) pipelines.

### Architecture
#### Frontend: 
Built using Flask templates (index.html and home.html) for user input and result visualization.
#### Backend: 
A Python Flask application handling API requests and serving predictions using pre-trained models.
#### Machine Learning:
    Models used:
    CatBoost
    K-Nearest Neighbors (KNN)
    LightGBM (LGBM)
    XGBoost (XGB)
    Ensemble of the above models.

### Feature Engineering 
Includes handling missing values, one-hot encoding, and scaling.
### Technologies Used
#### Languages: 
Python
#### Frameworks: 
Flask, Scikit-learn, LightGBM, XGBoost, CatBoost
#### Deployment: 
Docker, AWS ECS, Elastic Beanstalk, Azure Web Apps
#### CI/CD: 
GitHub Actions
#### Cloud Services: 
AWS (ECS, ECR, Beanstalk), Azure

### Application Flow

#### Input Features:

person_age
person_income
person_home_ownership
person_emp_length
loan_intent
loan_grade
loan_amnt
loan_int_rate
cb_person_default_on_file
cb_person_cred_hist_length
Feature Engineering:

Calculate loan_percent_income:
loan_percent_income
=
(
loan_amnt
person_income
)
×
100
loan_percent_income=( 
person_income
loan_amnt
​
 )×100

### Prediction Pipeline:

Input: User-provided data via the web interface.
Processing: Custom preprocessing pipeline.
Inference: Predict using an ensemble of pre-trained models.
Output:

Displays whether the applicant is likely to default.
Highlights predictions from individual models and the ensemble result.

### Model Performance:
[ 2024-11-13 01:18:25,596 ] 117 root - INFO - knn - Accuracy: 0.8882921589688507, Precision: 0.8125544899738448, Recall: 0.6449826989619377, Specificity: 0.9576104100946372
[ 2024-11-13 01:18:26,176 ] 117 root - INFO - xgb - Accuracy: 0.9338652754334816, Precision: 0.9592391304347826, Recall: 0.7328719723183391, Specificity: 0.9911277602523659
[ 2024-11-13 01:18:27,941 ] 117 root - INFO - cat - Accuracy: 0.9357066134724566, Precision: 0.9613309352517986, Recall: 0.7397923875432526, Specificity: 0.9915220820189274
[ 2024-11-13 01:18:28,238 ] 117 root - INFO - lgb - Accuracy: 0.9337118305969004, Precision: 0.9634034766697164, Recall: 0.728719723183391, Specificity: 0.9921135646687698
[ 2024-11-13 01:18:33,047 ] 117 root - INFO - ensemble - Accuracy: 0.9353997237992941, Precision: 0.9714548802946593, Recall: 0.7301038062283737, Specificity: 0.9938880126182965

### License
This project is licensed under the MIT License.

### Acknowledgements
Peers and Krish-Naik's youtube videos for E2E deployment.

