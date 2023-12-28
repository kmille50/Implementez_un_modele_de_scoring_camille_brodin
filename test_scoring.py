
import os
#from scoring import function
import pytest
import pandas as pd
from sklearn import metrics
from sklearn.metrics import fbeta_score, make_scorer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, TargetEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
import pickle
import random

X_train = pd.read_csv('X_train.csv') 
y_train = pd.read_csv('y_train.csv')  
test = pd.read_csv('df.csv') 
df_final = pd.read_csv('df_final.csv')
    
#test numéro 1 : métrique banquaire 
def bank_metric(y_true, y_pred):
    return fbeta_score(y_true, y_pred, beta=10)    
    
def test_bank_metric():
    assert bank_metric(y_train, y_train) == 1
    
    
    
#test numéro 2 : seuil solvabilité     
def transform_treshold(df, tresh):
    new_df = (df >= tresh).astype("int")
    return new_df

def test_transform_treshold():
    assert transform_treshold(df_final['y_proba'].iloc[1], 0.25) == 1
    
    

#test numéro 3: prédiction
pkl_filename = "model.pkl"
with open(pkl_filename, 'rb') as f_in:
    model = pickle.load(f_in)

def predict_bank(X, model):
    model.fit(X_train, y_train)
    y_proba = model.predict_proba(X)[:, 1]
    y_pred = (y_proba > 0.25).astype("int")
    return y_pred

def test_predict_bank():
    assert predict_bank(test, model) == 1