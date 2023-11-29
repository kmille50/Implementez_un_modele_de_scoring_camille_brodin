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

app = Flask(__name__)

model = joblib.load("gb_final_model.pkl")
mlflow.pyfunc.load_model("runs:/a13e39e874f447b49357cd9ce038968b/model")
THRESHOLD = 0.25

@app.post('/predict')
def predict():
    params =request.params
    df=pd.Series(params)
    proba = model.predict_proba(df)
    result = 1 if proba > THRESHOLD else 0
    record = {'result' : result}
    return jsonify(record)