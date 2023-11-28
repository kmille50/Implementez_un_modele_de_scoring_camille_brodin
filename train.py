import pandas as pd
import numpy as np
import json
import joblib
import ModelWrapper
from flask import Flask, jsonify, request, jsonify, render_template
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.compose import ColumnTransformer
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import TargetEncoder, OneHotEncoder, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn import set_config
import mlflow
mlflow.autolog()

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
  
class ModelWrapper:

    def __init__(self, model: Pipeline, threshold: float):
        self.pipeline = Pipeline
        self.threshold = threshold

    def fit(self, X_train, y_train):
        return self.pipeline.fit(X_train, y_train)

    def predict(self, X) -> int:
        proba = self.pipeline.predict_proba(X)
        result = 1 if proba > self.threshold else 0
        return 0

    # def score(self, y_pred, y_test) -> float:
    #     return bank_metric(y_pred, y_test)
    # def bank_metric(self, y_test, y_pred):
    #     return fbeta_score(y_test, y_pred, beta=10)
    #     bank_score = make_scorer(bank_metric, greater_is_better=True)

my_pipeline = ModelWrapper(GradientBoosting_tuned)
joblib.dump(my_pipeline, 'gb_final_model.pkl')

signature = my_pipeline._model_meta._signature
model = mlflow.pyfunc.save_model(my_pipeline, 'gb_final_model', signature = signature)
