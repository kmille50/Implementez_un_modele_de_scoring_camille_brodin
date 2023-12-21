import flask
import pandas as pd
import numpy as np
import json
import os
from flask import Flask, jsonify, request, render_template, redirect, Request
from sklearn.ensemble import GradientBoostingClassifier
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import TargetEncoder, OneHotEncoder, MinMaxScaler
from sklearn.impute import SimpleImputer
import pickle

app = Flask(__name__)
app.config["DEBUG"] = True

# Load from file
pkl_filename = "model.pkl"
with open(pkl_filename, 'rb') as f_in:
    model = pickle.load(f_in)
    
@app.route("/predict", methods=["POST"])
def predict():
    data = request.json #json -> dictionnaire
    df = pd.Series(data).to_frame().T #dictionnaire -> series -> dataframe bonne orientation
    probas = pd.DataFrame(model.predict_proba(df)) #predict numpyarray -> dataframe
    preds = (probas > 0.25).astype("int")
    proba = round(probas.values.tolist()[0][0]*100,2)
    pred = preds.values.tolist()[0][0] #numpyarray -> list et sélection premier element
    response = {"Probabilité de défaut de remboursement (%)" : proba, "Crédit_accordé(0)_ou_refusé(1)" : pred} #dictionnaire python
    return jsonify(response) #dictionnaire-> http json

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8000)