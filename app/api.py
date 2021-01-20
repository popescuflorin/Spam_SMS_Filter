# Dependencies
from flask import Flask, request, jsonify
import joblib
import traceback
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import json
from flask_cors import CORS
# Your API definition
app = Flask(__name__)
cors = CORS(app, resources={r"/*": {"origins": "*"}})
@app.route('/predict', methods=['OPTIONS'])
def option():
    return "200", 200


@app.route('/predict', methods=['POST'])
def predict():
    if lr:
        try:
            print("_____________________________")
            json_ = request.json
            sms = json_[0]["message"]
            print(sms)
            xplm = sc.transform([sms])
            print(type(xplm))
            return jsonify({'prediction': str(lr.predict(xplm))})

        except:

            return jsonify({'trace': traceback.format_exc()})
    else:
        print ('Train the model first')
        return ('No model here to use')

if __name__ == '__main__':
    lr = joblib.load("model.pkl") # Load "model.pkl"
    sc = joblib.load("cv.pkl") # Load "model.pkl"
    print ('Model loaded')
    #model_columns = joblib.load("model_columns.pkl") # Load "model_columns.pkl"

    app.run(debug=True)