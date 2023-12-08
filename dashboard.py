import streamlit as st
import pandas as pd
import numpy as np


st.title('Prêt à dépenser : Dashboard solvabilité client')

url_api = "http://localhost:5000/"

age = st.text_input('DAYS_BIRTH')
output = st.text_input(range(1,70,1))

# def display():
#     params = {"DAYS_BIRTH": age.value}
#     response = requests.post(url, params)
#     record = response.json()
#     result = record['result']
#     output.set_value(result)

# if __name__ == "__main__":
#     main()
