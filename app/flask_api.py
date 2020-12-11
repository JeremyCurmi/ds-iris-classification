from flask import Flask, request

import joblib
import pandas as pd
import numpy as np

app=Flask(__name__) # from which point you want to start the app

joblib_in = open('model.pkl','rb')
classifier = joblib.load(joblib_in)

@app.route('/') # decorater # this is my route api
def welcome():
    return "Welcome All"

@app.route('/predict', methods=["GET"]) # when method is not specified, automatically its set as GET
def predict_iris_species():
    sepal_length = float(request.args.get('sepal_length'))
    sepal_width = float(request.args.get('sepal_width'))
    petal_length = float(request.args.get('petal_length'))
    petal_width = float(request.args.get('petal_width'))

    print(f"sepal_length: {sepal_length}")
    print(f"sepal_width: {sepal_width}")
    print(f"petal_length: {petal_length}")
    print(f"petal_width: {petal_width}")
    prediction = classifier.predict([[sepal_length,sepal_width,petal_length,petal_width]])

    return f"The predicted value is {prediction}"

@app.route('/predict_file', methods=["POST"]) # Post was required as we need to give in a file
def predict_file_iris_species():
    print("loading data file ...")
    df = pd.read_csv(request.files.get("file"))
    prediction = classifier.predict(df)

    return f"The predicted value for the csv {list(prediction)}"


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)


