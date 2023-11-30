from joblib import load

from flask import Flask, jsonify, request, jsonify, render_template
import json

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import NearestNeighbors
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

# Flask route decorators map / and /hello to the hello function.
# To add other resources, create functions that generate the page contents
# and add decorators to define the appropriate resource locators for them.

# On charge les données
data_train = pd.read_csv("X_train.csv")
data_test = pd.read_csv("X_test.csv")
model = joblib.load("gb_final_model.pkl")
mlflow.pyfunc.load_model("runs:/cc75ed32247d4124ad24aba20f2a951e/model")
THRESHOLD = 0.25

# On crée deux variables en attente qui deviendront
# des variables globales après l'initialisation de l'API.
# Ces variables sont utilisées dans plusieurs fonctions de l'API.
train = None
test = None
model = None

# On crée la liste des ID clients qui nous servira dans l'API
id_client = data_test["SK_ID_CURR"][:50].values
id_client = pd.DataFrame(id_client)

# routes
# Entraînement du modèle
@app.route("/init_model", methods=["GET"])
def init_model():

    # On transforme le dataset de test préparé en variabe
    # globale, car il est utilisé dans la fonction predict
    global train
    train = df_train.copy()

    global test
    test = df_test.copy()

    # On entraîne le modèle et on le transforme en
    # variable globale pour la fonction predict
    global clf_gb
    clf_gb = entrainement_pipeline(X, y)
    print("Training pipeline done")

    return jsonify(["Initialisation terminée."])


# Chargement des données pour la selection de l'ID client
@app.route("/load_data", methods=["GET"])
def load_data():
    
    return id_client.to_json(orient='values')

# Chargement d'informations générales
@app.route("/infos_gen", methods=["GET"])
def infos_gen():

    lst_infos = [data_train.shape[0],
                 round(data_train["AMT_INCOME_TOTAL"].mean(), 2),
                 round(data_train["AMT_CREDIT"].mean(), 2)]

    return jsonify(lst_infos)

# Chargement des données pour le graphique
# dans la sidebar
@app.route("/disparite_target", methods=["GET"])
def disparite_target():

    df_target = data_train["TARGET"].value_counts()

    return df_target.to_json(orient='values')

# Chargement d'informations générales sur le client
@app.route("/infos_client", methods=["GET"])
def infos_client():

    id = request.args.get("id_client")

    data_client = data_test[data_test["SK_ID_CURR"] == int(id)]

    print(data_client)
    dict_infos = {
       "status_famille" : data_client["NAME_FAMILY_STATUS"].item(),
       "nb_enfant" : data_client["CNT_CHILDREN"].item(),
       "age" : int(data_client["DAYS_BIRTH"].values / -365),
       "revenus" : data_client["AMT_INCOME_TOTAL"].item(),
       "montant_credit" : data_client["AMT_CREDIT"].item(),
       "annuites" : data_client["AMT_ANNUITY"].item(),
       "montant_bien" : data_client["AMT_GOODS_PRICE"].item()
       }
    
    response = json.loads(data_client.to_json(orient='index'))

    return response

# Calcul des ages de la population pour le graphique
# situant l'age du client
@app.route("/load_age_population", methods=["GET"])
def load_age_population():
    
    df_age = round((data_train["DAYS_BIRTH"] / -365), 2)
    return df_age.to_json(orient='values')

# Segmentation des revenus de la population pour le graphique
# situant l'age du client
@app.route("/load_revenus_population", methods=["GET"])
def load_revenus_population():
    
    # On supprime les outliers qui faussent le graphique de sortie
    # Cette opération supprime un peu moins de 300 lignes sur une
    # population > 300000...
    df_revenus = data_train[data_train["AMT_INCOME_TOTAL"] < 700000]
    
    df_revenus["tranches_revenus"] = pd.cut(df_revenus["AMT_INCOME_TOTAL"], bins=20)
    df_revenus = df_revenus[["AMT_INCOME_TOTAL", "tranches_revenus"]]
    df_revenus.sort_values(by="AMT_INCOME_TOTAL", inplace=True)

    print(df_revenus)
    
    df_revenus = df_revenus["AMT_INCOME_TOTAL"]

    return df_revenus.to_json(orient='values')

@app.route("/predict", methods=["GET"])
def predict():
    
    id = request.args.get("id_client")

    print("Analyse data_test :")
    print(data_test.shape)
    print(data_test[data_test["SK_ID_CURR"] == int(id)])
     
    index = data_test[data_test["SK_ID_CURR"] == int(id)].index.values

    print(index[0])
   
    data_client = test[index]

    print(data_client)

    prediction = pd.DataFrame(clf_gb.predict_proba(data_client))
    prediction = (prediction >= THRESHOLD).astype("int")
    prediction = prediction[0].tolist()

    print(prediction)

    return jsonify(prediction)

@app.route("/load_voisins", methods=["GET"])
def load_voisins():
    
    id = request.args.get("id_client")

    index = data_test[data_test["SK_ID_CURR"] == int(id)].index.values

    data_client = test[index]
    
    distances, indices = knn.kneighbors(data_client)

    print("indices")
    print(indices)
    print("distances")
    print(distances)

    df_voisins = data_train.iloc[indices[0], :]
    
    response = json.loads(df_voisins.to_json(orient='index'))

    return response

def entrainement_pipeline(X, y):

    # Configuration de la meilleure itération trouvée par le RandomizeSearchCV
    # Optimized n_estimator=1144
    clf_gb = model.fit(X, y)

    return clf_gb


if __name__ == "__main__":
    app.run(host="localhost", port="5000", debug=True)