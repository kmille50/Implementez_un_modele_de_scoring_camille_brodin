import pandas as pd
import numpy as np
import json
import joblib
from flask import Flask, jsonify, request, jsonify, render_template
from sklearn.ensemble import GradientBoostingClassifier
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import TargetEncoder, OneHotEncoder, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.externals import joblib
import mlflow
mlflow.autolog()
  
class ModelWrapper:

    def __init__(self, model: Pipeline, threshold: float = 0.25):
        self.pipeline = pipeline
        self.threshold = threshold

    def predict(self, X) -> int:
        proba = self.pipeline.predict_proba(X)
        result = 1 if proba > self.threshold else 0
        return 0

my_pipeline = ModelWrapper(GradientBoosting_tuned)
joblib.dump(my_pipeline, 'gb_final_model.pkl')

signature = model._model_meta._signature
model = mlflow.pyfunc.save_model(GradientBoosting_tuned, 'GradientBoosting_bank_test', signature = signature)
