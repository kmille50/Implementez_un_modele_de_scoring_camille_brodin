from flask import Flask, request, redirect
import os
import jsonify
#from prediction import predict_bank

app = Flask(__name__)

pkl_filename = "model.pkl"
with open(pkl_filename, 'rb') as f_in:
    model = pickle.load(f_in)

@app.post('/predict')
def predict_bank(request):
    ##loading the model from the saved file

    params = request.params
    df = pd.Series(params)
    y_proba = model.predict_proba(df)[:, 1]
    y_pred = (y_proba > 0.25).astype("int")
    record = {'y_proba': y_proba, 'y_pred': y_pred}
    
    return jsonify(record)

if __name__ == "__main__":
    app.run(host="localhost", port="5000", debug=True)