import streamlit as st
import requests

#url = 'http://127.0.0.1:5000'

# age = st.text_input('DAYS_BIRTH')
# output = st.text_area(range(1,70,1))

# def display():
#     params = {"DAYS_BIRTH": age.value}
#     response = requests.post(url, params)
#     record = response.json()
#     result = record['result']
#     output.set_value(result)
    

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
    MLFLOW_URI =  'http://127.0.0.1:5000'
    
    api_choice = st.sidebar.selectbox(
        'Quelle API souhaitez vous utiliser',
        ['MLflow'])
    
    st.title('Prediction de solvabilité client')

    DAYS_BIRTH = st.number_input('Age de lindividu',
                                 min_value=0., value=70., step=1.)

    AMT_INCOME_TOTAL  = st.number_input('Revenu totaux',
                              min_value=0., value=4500000., step=500.)

    CNT_CHILDREN = st.number_input('Nombre denfant',
                                   min_value=0., value=14., step=1.)

    OCCUPATION_TYPE = st.number_input('Type dactivite professionnelle',
                                     min_value=0., value=3., step=1.)

    CREDIT_INCOME_PERCENT = st.number_input('Taux dendettement',
                                min_value=0., value=-119., step=1.)

    predict_btn = st.button('Prédire')
    
    if predict_btn:
        data = [[DAYS_BIRTH, AMT_INCOME_TOTAL, CNT_CHILDREN,
                 OCCUPATION_TYPE, CREDIT_INCOME_PERCENT]]
        pred = request_prediction(MLFLOW_URI, data=data)[0] * 100000

        st.write(
            'La probabilité de faillite de ce client est de {:.2f}'.format(pred))


if __name__ == '__main__':
    main()