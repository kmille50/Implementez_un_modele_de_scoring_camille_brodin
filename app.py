import flask
import pandas as pd
import numpy as np
import json
import joblib
import os
from flask import Flask, jsonify, request, jsonify, render_template, redirect, Request
from sklearn.ensemble import GradientBoostingClassifier
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import TargetEncoder, OneHotEncoder, MinMaxScaler
from sklearn.impute import SimpleImputer
from prediction import predict_bank
import pickle
import mlflow

# Create an instance of the Flask class that is the WSGI application.
# The first argument is the name of the application module or package,
# typically __name__ when using a single module.

app = Flask(__name__)
app.config["DEBUG"] = True

X_tr = pd.read_csv('X_tr.csv')
X_te = pd.read_csv('X_te.csv')
y_tr = pd.read_csv('y_tr.csv')
y_te = pd.read_csv('y_te.csv')

#X_tr = X_tr.drop(columns="TARGET")
#X_te = X_te.drop(columns="TARGET")
#y_tr = y_tr.drop(columns="TARGET")
#y_te = y_te.drop(columns="TARGET")

# Load from file
pkl_filename = "model.pkl"
with open(pkl_filename, 'rb') as f_in:
    model = pickle.load(f_in)

# On crée la liste des ID clients qui nous servira dans l'API
id_client = X_te["SK_ID_CURR"][:50].values
id_client = pd.DataFrame(id_client)

# Chargement des données pour la selection de l'ID client
@app.route("/load_data", methods=["POST"])
def load_data():
   return id_client.to_json(orient='values')

# Chargement d'informations générales
@app.route("/infos_gen", methods=["POST"])
def infos_gen():

    lst_infos = [X_tr.shape[0],
                 round(X_tr["AMT_INCOME_TOTAL"].mean(), 2),
                 round(X_tr["AMT_CREDIT"].mean(), 2)]

    return jsonify(lst_infos)

# Chargement des données pour le graphique
# dans la sidebar
@app.route("/disparite_target", methods=["POST"])
def disparite_target():

    df_target = X_tr["TARGET"].value_counts()

    return df_target.to_json(orient='values')


# Chargement d'informations générales sur le client
# @app.route("/infos_client", methods=["POST"])
# def infos_client():
    
    #id = request.args.get("id_client")
    #data_client = X_te[x_te["SK_ID_CURR"] == int(id)]
    
    # dict_infos = {
    #    "status_famille" : data_client["NAME_FAMILY_STATUS"].item(),
    #    "nb_enfant" : data_client["CNT_CHILDREN"].item(),
    #    "age" : int(data_client["DAYS_BIRTH"].values / -365),
    #    "revenus" : data_client["AMT_INCOME_TOTAL"].item(),
    #    "montant_credit" : data_client["AMT_CREDIT"].item(),
    #    "annuites" : data_client["AMT_ANNUITY"].item(),
    #    "montant_bien" : data_client["AMT_GOODS_PRICE"].item()
    #    }
    
    # response = json.loads(data_client.to_json(orient='index'))

    # return response


# @app.route("/predict_bank", methods=["POST"])
# def predict():
#     params = request.get(url = "http://127.0.0.1:5000/", params=params)
#     df = pd.Series(params)
#     predict = prediction.predict_bank(df, model)
#     predictOutput = {'predict': predict}
#     return jsonify(predictOutput)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)