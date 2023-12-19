import streamlit as st
import pandas as pd
import numpy as np
import api
import requests
import json
import numpy as np
import plotly.figure_factory as ff

url = "http://127.0.0.1:5000/predict"

header = st.container()
dataset = st.container()
features = st.container()
model_training = st.container()

with header:
       st.title("Prêt à dépenser : Solvabilité client")
       st.text("Dashboard interactif pour plus de transparence sur les décisions d'octroi de crédit, et mise à disposition des clients leurs informations personnelles avec exploration facilité.")

with dataset:
       st.header("Déja client ? Consultez vos données en toute transparence!")
       st.text("Ce dataset est disponible a cette adresse: https://www.kaggle.com/competitions/home-credit-default-risk/data")
       clients_data = pd.read_csv('X_tr.csv')     
       
       id = st.selectbox("Identifiant client:", clients_data['SK_ID_CURR'].unique())
       st.write(clients_data.loc[clients_data['SK_ID_CURR'] == id])
       
       st.subheader("Distribution des annuités de prêt")
       annuite = pd.DataFrame(clients_data['CREDIT_INCOME_PERCENT'].value_counts()).head()
       st.bar_chart(annuite)
       
       st.subheader("Distribution du type des contrats de crédit")
       type_credit = pd.DataFrame(clients_data['TARGET'].value_counts()).head()
       st.bar_chart(type_credit)
       
       
with model_training:

       st.header('Vous réfléchissez à prendre un crédit ? Faites une simulation dès maintenant!')
       st.text("Ici vous pouvez tester des informations clients pour générer une prédiction de solvabilité:")        
       
       DAYS_BIRTH = st.number_input('DAYS_BIRTH', min_value=21., max_value=70., value=44., step=1.)
       DAYS_EMPLOYED = st.number_input('DAYS_EMPLOYED', min_value=0., max_value=18000., value=2395., step=1.)
       CREDIT_TERM = st.number_input('CREDIT_TERM', min_value=0., max_value = 1., value=0.05, step=0.1)
       CREDIT_INCOME_PERCENT = st.number_input('CREDIT_INCOME_PERCENT', min_value=0., max_value =40., value=4., step=1.)
       EXT_SOURCE_1 = st.number_input('EXT_SOURCE_1', min_value=0., max_value =1., value=0.5, step=0.1)
       EXT_SOURCE_2 = st.number_input('EXT_SOURCE_2', min_value=0., max_value =1., value=0.5, step=0.1)
       EXT_SOURCE_3 = st.number_input('EXT_SOURCE_3', min_value =0., max_value =1., value=0.5, step=0.1)
       ORGANIZATION_TYPE = st.selectbox('ORGANIZATION_TYPE', clients_data['ORGANIZATION_TYPE'].unique())
       OCCUPATION_TYPE = st.selectbox(' OCCUPATION_TYPE', clients_data['OCCUPATION_TYPE'].unique())
       CNT_CHILDREN = st.number_input('CNT_CHILDREN', min_value =0., max_value =10., value=0., step=1.)
       

       if 'prediction' not in st.session_state:
              st.session_state.prediction = {}

       def request_model():
              params = {
                    "DAYS_BIRTH": DAYS_BIRTH, "DAYS_EMPLOYED": DAYS_EMPLOYED, "CREDIT_TERM" : CREDIT_TERM,
                    "CREDIT_INCOME_PERCENT" : CREDIT_INCOME_PERCENT, "EXT_SOURCE_1" : EXT_SOURCE_1, "EXT_SOURCE_2" : EXT_SOURCE_2,
                    "EXT_SOURCE_3" : EXT_SOURCE_3, "ORGANIZATION_TYPE" : ORGANIZATION_TYPE, "OCCUPATION_TYPE" : OCCUPATION_TYPE,
                    "CNT_CHILDREN" : CNT_CHILDREN
                    }
              
              response = requests.post(url, json=params)
              response.status_code == 200
              st.session_state.prediction = response.json()
              
       
       st.button('Click me', on_click=request_model)

       if st.session_state.prediction:
              st.write(st.session_state.prediction)
