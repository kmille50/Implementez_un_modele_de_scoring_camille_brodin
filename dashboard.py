import streamlit as st
import requests

url = 'http://127.0.0.1:5000'

def main():
   
    st.title('Prediction de solvabilité client')

    DAYS_BIRTH = st.number_input('Age de lindividu',
                                 min_value=0., value=70., step=1.)
    
    CODE_GENDER  = st.text_input('Sexe de lindividu',
                              value=("Homme","Femme"))

    AMT_INCOME_TOTAL  = st.number_input('Revenu totaux',
                              min_value=0., value=4500000., step=500.)

    CNT_CHILDREN = st.number_input('Nombre denfant',
                                   min_value=0., value=14., step=1.)

    OCCUPATION_TYPE = st.number_input('Type dactivite professionnelle',
                                     min_value=0., value=3., step=1.)

    DAYS_LAST_PHONE_CHANGE = st.number_input('Jours depuis le dernier changement de téléphone ',
                                 min_value=0., value=1425, step=100)

    CREDIT_INCOME_PERCENT = st.number_input('Taux dendettement',
                                min_value=0., value=-119., step=1.)

    predict_btn = st.button('Prédire')
    
    if predict_btn:
        data = [[DAYS_BIRTH, CODE_GENDER, AMT_INCOME_TOTAL, CNT_CHILDREN,
                 OCCUPATION_TYPE, DAYS_LAST_PHONE_CHANGE, CREDIT_INCOME_PERCENT]]
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