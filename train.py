import pandas as pd
import numpy as np
import json
import joblib
import os
import mlflow
mlflow.autolog()
from flask import Flask, jsonify, request, jsonify, render_template
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.compose import ColumnTransformer
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import TargetEncoder, OneHotEncoder, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import make_scorer, fbeta_score
from sklearn import set_config
from mlflow.models import infer_signature

# On charge les données
X_train = pd.read_csv("X_train.csv")
X_test = pd.read_csv("X_test.csv")
y_train = pd.read_csv("y_train.csv")
y_test = pd.read_csv("y_test.csv")

numerical_features = X_train.select_dtypes(exclude="object").columns
categorical_features = X_train.select_dtypes(include='object').columns
features_X_train = X_train.columns
features_X_test = X_test.columns

# Define the encoding strategy for features
numerical_features = X_train.select_dtypes(exclude="object").columns
n_unique_categories = X_train[categorical_features].nunique().sort_values(ascending=False)
high_cardinality_features = n_unique_categories[n_unique_categories > 15].index
low_cardinality_features = n_unique_categories[n_unique_categories <= 15].index

set_config(transform_output="pandas")

# Define the preprocessing steps for numeric features
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', MinMaxScaler())])

# Define the preprocessing steps for categorical features
high_categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('targ', TargetEncoder()),
    ('scaler', MinMaxScaler())])

# Define the preprocessing steps for categorical features
low_categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(sparse_output=False, handle_unknown="ignore"))])

# Use ColumnTransformer to apply the transformations to the correct columns in the dataframe
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numerical_features),
        ('high_cat', high_categorical_transformer, high_cardinality_features),
        ('low_cat', low_categorical_transformer, low_cardinality_features)])

steps = [('preprocessor', preprocessor), ('over', SMOTE()), 
          ('model',  GradientBoostingClassifier(learning_rate= 0.1, n_estimators= 50, random_state=1))]
GradientBoosting_tuned = Pipeline(steps=steps)

mlflow.start_run()
run = mlflow.active_run()
  
class ModelWrapper:

    def __init__(self, model: Pipeline, threshold: float=0.25):
        self.pipeline = Pipeline
        self.threshold = 0.25

    def fit(self, X_train, y_train):
        return self.pipeline.fit(X_train, y_train)

    def predict(self, X) -> int:
        proba = pd.DataFrame(self.pipeline.predict_proba(X))
        y_pred = (proba >= seuil).astype("int")
        return y_pred
 
my_pipeline = ModelWrapper(model=GradientBoosting_tuned, threshold=0.25)
joblib.dump(my_pipeline, 'gb_final_model.pkl')

#from mlflow.models.signature import infer_signature
#model = mlflow.pyfunc.load_model("runs:/2d8c6ace67f545eaae0a6b265dcbd010/model")
#signature = model._model_meta._signature
#mlflow.sklearn.save_model(my_pipeline, 'mlflow_model', signature=signature)
mlflow.end_run()

#signature = infer_signature(X_train, my_pipeline.predict(X_train))
#mlflow.pyfunc.save_model(my_pipeline, 'gb_final_model_mlflow')
#logged_model = 'runs:/2d8c6ace67f545eaae0a6b265dcbd010/model'

# Load model.
#loaded_model = mlflow.sklearn.load_model(logged_model)