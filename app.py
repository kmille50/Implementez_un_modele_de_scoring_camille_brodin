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
import requests

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


#Chargement d'informations générales sur le client
@app.route("/infos_client", methods=["POST"])
def infos_client():
    
    id = request.args.get("id_client")
    data_client = X_te.iloc[:50]
    #data_client = X_te[X_te["SK_ID_CURR"] == [id]]
    
    dict_infos = {
       "status_famille" : data_client["NAME_FAMILY_STATUS"].to_list(),
       "nb_enfant" : data_client["CNT_CHILDREN"].to_list(),
        "age" : round(data_client["DAYS_BIRTH"]/365, 2).to_list(),
       "revenus" : data_client["AMT_INCOME_TOTAL"].to_list(),
       "montant_credit" : data_client["AMT_CREDIT"].to_list(),
       "annuites" : data_client["AMT_ANNUITY"].to_list(),
       "montant_bien" : data_client["AMT_GOODS_PRICE"].to_list()
       }
    
    response = json.loads(data_client.to_json(orient='index'))

    return response

# Calcul des ages de la population pour le graphique
# situant l'age du client
@app.route("/load_age_population", methods=["POST"])
def load_age_population():
    df_age = round((X_tr["DAYS_BIRTH"] / 365), 2)
    return df_age.to_json(orient='values')

# Segmentation des revenus de la population pour le graphique
# situant l'age du client
@app.route("/load_revenus_population", methods=["POST"])
def load_revenus_population():
    
    # On supprime les outliers qui faussent le graphique de sortie
    # Cette opération supprime un peu moins de 300 lignes sur une
    # population > 300000...
    df_revenus = X_tr[X_tr["AMT_INCOME_TOTAL"] < 700000]
    
    df_revenus["tranches_revenus"] = pd.cut(df_revenus["AMT_INCOME_TOTAL"], bins=20)
    df_revenus = df_revenus[["AMT_INCOME_TOTAL", "tranches_revenus"]]
    df_revenus.sort_values(by="AMT_INCOME_TOTAL", inplace=True)

    print(df_revenus)
    
    df_revenus = df_revenus["AMT_INCOME_TOTAL"]

    return df_revenus.to_json(orient='values')

def predict_bank(X, model):
    
    if type(X) == dict:
        df = pd.DataFrame([X])
    else:
        df = pd.DataFrame(X).T
    
    model.fit(X_tr, y_tr)
    y_proba = model.predict_proba(df)[:, 1]
    y_pred = (y_proba > 0.25).astype("int")
    
    if y_pred == 0:
        print('Client solvable :', "la probabilité de faillite est de",y_proba*100, "%")
    elif y_pred == 1:
        print('Client à risque :', "la probabilité de faillite est de",y_proba*100, "%")
    
    return y_pred


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    predict = predict_bank(data, model)
    predictOutput = {'predict': predict}
    return jsonify(predictOutput)

# @app.route("/predict", methods=["POST"])
# def predict():
#     params = requests.get(url = "http://127.0.0.1:5000/infos_client", params=params)
#     df = pd.Series(params)
#     predict = prediction.predict_bank(df, model)
#     predictOutput = {'predict': predict}
#     return jsonify(predictOutput)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)