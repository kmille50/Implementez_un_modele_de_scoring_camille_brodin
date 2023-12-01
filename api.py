import flask
import pandas as pd
import numpy as np
import json
import joblib
from flask import Flask, jsonify, request, jsonify, render_template
from sklearn.ensemble import GradientBoostingClassifier
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import TargetEncoder, OneHotEncoder, MinMaxScaler
from sklearn.impute import SimpleImputer
import pickle
import mlflow

# Create an instance of the Flask class that is the WSGI application.
# The first argument is the name of the application module or package,
# typically __name__ when using a single module.

app = flask.Flask(__name__)
app.config["DEBUG"] = True

model = joblib.load("GradientBoosting_tuned.joblib")
model_labels = ['0','1']

@app.route('/predict', methods=['POST'])
def predict():
    # Retrieve query parameters related to this request.
    DAYS_BIRTH = request.args.get('DAYS_BIRTH')
    AMT_INCOME_TOTAL = request.args.get('AMT_INCOME_TOTAL')
    CNT_CHILDREN = request.args.get('CNT_CHILDREN')
    OCCUPATION_TYPE = request.args.get('OCCUPATION_TYPE')
    CREDIT_INCOME_PERCENT = request.args.get('CREDIT_INCOME_PERCENT')

    # Our model expects a list of records
    features = [[DAYS_BIRTH, AMT_INCOME_TOTAL, CNT_CHILDREN, OCCUPATION_TYPE, CREDIT_INCOME_PERCENT]]
    
    # Use the model to predict the class
    label_index = model.predict_proba(features)
    # Retrieve the iris name that is associated with the predicted class
    label = model_labels[label_index[0]]
    # Create and send a response to the API caller
    return jsonify(status='complete', label=label)

if __name__ == '__main__':
    app.run(debug=True)

