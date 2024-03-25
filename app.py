import numpy as np
import pandas as pd

from flask import Flask, request, render_template

from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

application = Flask(__name__) #Entry point

app = application

## Route for a home page (if needed)
@app.route('/')
def index():
    return render_template('index.html') 

## Get and predict the data
@app.route('/predict', methods=['GET','POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('index.html')
    else:
        data = CustomData(
            gender = request.form.get('gender'),
            race_ethnicity = request.form.get('ethnicity'),
            parental_level_of_education = request.form.get('parental_level_of_education'),
            lunch = request.form.get('lunch'),
            test_preparation_course = request.form.get('test_preparation_course'),
            reading_score = float(request.form.get('writing_score')),
            writing_score = float(request.form.get('reading_score'))

        )
        pred_df = data.get_data_as_data_frame()
        print(pred_df)
        print("Before Prediction")

        predict_pipeline = PredictPipeline()
        print("Mid Prediction")
        
        result = predict_pipeline.predict(pred_df)
        results = round(result[0], 2)
        print("After Prediction")
        
        return render_template('index.html', results = results)
    

if __name__=="__main__":
    app.run(host="0.0.0.0", port=8000)