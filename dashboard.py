import streamlit as st
import pandas as pd
import numpy as np
import api
import requests

url = "http://127.0.0.1:5000/predict"

header = st.container()
dataset = st.container()
features = st.container()
model_training = st.container()

output = st.text("")

with header:
       st.title("Prêt à dépenser : Solvabilité client")
       st.text("Dashboard interactif pour plus de transparence sur les décisions d'octroi de crédit, et mise à disposition des clients leurs informations personnelles avec exploration facilité.")

with dataset:
       st.header("Donnees informations clients")
       st.text("Ce dataset est disponible a cette adresse: https://www.kaggle.com/competitions/home-credit-default-risk/data")
       clients_data = pd.read_csv('X_train.csv')
       st.write(clients_data.head(3))
       
       st.subheader("Distribution du type des contrats de crédit")
       type_credit = pd.DataFrame(clients_data['NAME_CONTRACT_TYPE'].value_counts()).head()
       st.bar_chart(type_credit)
       
       st.subheader("Distribution des annuités de prêt")
       annuite = pd.DataFrame(clients_data['AMT_ANNUITY'].value_counts()).head()
       st.bar_chart(annuite)
       
with model_training:
       
       def request_prediction(model_uri, data):
              headers = {"Content-Type": "application/json"}

              data_json = {'data': data}
              response = requests.request(
              method='POST', headers=headers, url=model_uri, json=data_json)

              if response.status_code != 200:
                     raise Exception("Request failed with status {}, {}".format(response.status_code, response.text))

              return response.json()

       st.header('Entrainons le modèle de prédiction!')
       st.text("Ici vous pouvez tester des informations clients pour générer une prédiction de solvabilité:")
                     
       sel_col, disp_col = st.columns(2)
                     
       annuite_val = st.number_input("Quel montant d'annuité de prêt souhaiteriez vous?", min_value = 2000, max_value = 225000, value=24930, step=500)
                     
              #type_credit_val = sel_col.selectbox("Quel type de contrat souhaiteriez vous?", clients_data['NAME_CONTRACT_TYPE'].unique())
              # annuite_val = sel_col.slider("Quel montant d'annuité de prêt souhaiteriez vous?", min_value = 2000, max_value = 225000, value=24930, step=500)
              # #input_feature = sel_col.text_input("Quelle variable d'entrée ?", "")
                     
       predict_btn = st.button('Prédire')
       if predict_btn:
              data = [[annuite_val]]

              pred = request_prediction(url, data)[0]
              st.write('Le prix médian d\'une habitation est de {:.2f}'.format(pred))

       