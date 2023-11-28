import pandas as pd
import streamlit as st
import requests

def request_prediction(model_uri, data):
    headers = {"Content-Type": "application/json"}

    data_json = {'data': data}
    response = requests.request(
        method='POST', headers=headers, url=model_uri, json=data_json)

    if response.status_code != 200:
        raise Exception(
            "Request failed with status {}, {}".format(response.status_code, response.text))

    return response.json()


def main():
    MLFLOW_URI = 'http://127.0.0.1:5000'
    #CORTEX_URI = 'http://0.0.0.0:8890/'
    #RAY_SERVE_URI = 'http://127.0.0.1:8000/regressor'

    api_choice = st.sidebar.selectbox(
        'Quelle API souhaitez vous utiliser',
        ['MLflow']) #'Cortex', 'Ray Serve'

    st.title('Solvability Prediction')

    DAYS_BIRTH = st.number_input('Age de lindividu',
                                 min_value=0., value=3.87, step=1.)
    
    CODE_GENDER  = st.number_input('Sexe de lindividu',
                              min_value=0., value=28., step=1.)

    AMT_INCOME_TOTAL  = st.number_input('Revenu totaux',
                              min_value=0., value=28., step=1.)

    CNT_CHILDREN = st.number_input('Nombre denfant',
                                   min_value=0., value=5., step=1.)

    OCCUPATION_TYPE = st.number_input('Type dactivite professionnelle',
                                     min_value=0., value=3., step=1.)

    DAYS_LAST_PHONE_CHANGE = st.number_input('Jours depuis le dernier changement de téléphone ',
                                 min_value=0, value=1425, step=100)
    
    DAYS_EMPLOYED_PERCENT = st.number_input('Pourcentage de jour travaillé',
                               min_value=0., value=35., step=1.)

    CREDIT_INCOME_PERCENT = st.number_input('Taux dendettement',
                                min_value=0., value=-119., step=1.)

    predict_btn = st.button('Prédire')
    
    if predict_btn:
        data = [[DAYS_BIRTH, CODE_GENDER, AMT_INCOME_TOTAL, CNT_CHILDREN,
                 OCCUPATION_TYPE, DAYS_LAST_PHONE_CHANGE, DAYS_EMPLOYED_PERCENT, CREDIT_INCOME_PERCENT]]
        pred = None

        if api_choice == 'MLflow':
            pred = request_prediction(MLFLOW_URI, data)[0] * 100000
        #elif api_choice == 'Cortex':
        #    pred = request_prediction(CORTEX_URI, data)[0] * 100000
        #elif api_choice == 'Ray Serve':
        #    pred = request_prediction(RAY_SERVE_URI, data)[0] * 100000
        
        st.write(
            'La probabilité de faillite de ce client est de {:.2f}'.format(pred))


if __name__ == '__main__':
    main()