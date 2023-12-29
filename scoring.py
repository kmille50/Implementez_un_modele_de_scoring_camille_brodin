#Fonction projet 7

# Création d'une métrique adaptée Fbeta score pour le GridSearchCV https://scikit-learn.org/stable/modules/model_evaluation.html
from sklearn import metrics
from sklearn.metrics import fbeta_score, make_scorer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, TargetEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
import pickle
import pandas as pd  

def bank_metric(y_true, y_pred):
    return fbeta_score(y_true, y_pred, beta=10)

def transform_treshold(df, tresh):
    new_df = (df >= tresh).astype("int")
    return new_df


