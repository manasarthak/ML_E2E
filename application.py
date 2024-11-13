from flask import Flask, request, render_template
import numpy as np
import pandas as pd
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

application = Flask(__name__)
app = application

# Route for home page
@app.route('/')
def index():
    return render_template('index.html')

# Route for prediction
@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html')
    else:
        # Capture form data
        person_age = int(request.form.get('person_age'))
        person_income = float(request.form.get('person_income'))
        person_home_ownership = request.form.get('person_home_ownership')
        person_emp_length = float(request.form.get('person_emp_length'))
        loan_intent = request.form.get('loan_intent')
        loan_grade = request.form.get('loan_grade')
        loan_amnt = float(request.form.get('loan_amnt'))
        loan_int_rate = float(request.form.get('loan_int_rate'))
        cb_person_default_on_file = request.form.get('cb_person_default_on_file')
        cb_person_cred_hist_length = int(request.form.get('cb_person_cred_hist_length'))
        
        # Calculate "Loan Percent of Income"
        loan_percent_income = (loan_amnt / person_income) * 100

        # Create a CustomData object using these values
        data = CustomData(
            person_age=person_age,
            person_income=person_income,
            person_home_ownership=person_home_ownership,
            person_emp_length=person_emp_length,
            loan_intent=loan_intent,
            loan_grade=loan_grade,
            loan_amnt=loan_amnt,
            loan_int_rate=loan_int_rate,
            loan_percent_income=loan_percent_income,
            cb_person_default_on_file=cb_person_default_on_file,
            cb_person_cred_hist_length=cb_person_cred_hist_length
        )
        
        # Convert the data to a DataFrame for the prediction pipeline
        pred_df = data.get_data_as_dataframe()
        print("Input Data for Prediction:", pred_df)
        
        # Use the prediction pipeline to detect anomalies
        predict_pipe = PredictPipeline()
        results = predict_pipe.predict(pred_df)
        
        # Display each model's output and combined results in home.html
        return render_template('home.html', results=results)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5001, debug=True)
