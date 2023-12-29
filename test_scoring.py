
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

y_train = pd.read_csv('y_train.csv')  
df = pd.DataFrame()
df['x'] = 0.24, 0.26
    
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
    assert transform_treshold(df[1], 0.25) == 1
    
