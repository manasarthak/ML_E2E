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
        # Render home.html for data input
        return render_template('home.html')
    else:
        # Capture form data and create a CustomData object
        data = CustomData(
            transaction_amount=float(request.form.get('transaction_amount')),
            customer_age=int(request.form.get('customer_age'))
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
    app.run(host='127.0.0.1', port=5001, debug=True)

