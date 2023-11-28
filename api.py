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

# Create an instance of the Flask class that is the WSGI application.
# The first argument is the name of the application module or package,
# typically __name__ when using a single module.
app = Flask(__name__)

model = joblib.load("gb_final_model.pkl")

# model = mlflow.pyfunc.load_model("runs:/0a130dd1f0fa4ae7a84b625e06037ac4/model")

@app.post('/predict')
def predict(request):
    
    params = request.params
    df = pd.Series(params)
    result = model.predict(df)
    record = {"result" : result}
    return jsonify(record)


# if __name__ == "__main__":
#     app.run(host="localhost", port="5000", debug=True)