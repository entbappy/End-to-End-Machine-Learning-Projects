# Install flask, sklearn, pandas, pickle
from flask import Flask, render_template, request
import pandas as pd
import pickle
import numpy as np

app = Flask(__name__)

#Reading the cleaned data
data = pd.read_csv("Cleaned_data.csv")
pipe = pickle.load(open("Ridge_model.pkl","rb"))


@app.route("/")

def index():
    location = sorted(data['location'].unique())
    
    return render_template("index.html", locations = location)


@app.route('/predict',methods=['POST'])

def predict():

    location = request.form.get('location')
    bhk = float(request.form.get('bhk'))
    bath = float(request.form.get('bath'))
    sqft = float(request.form.get('total_sqft'))

    # As a dataframe we have to give our model
    inputs = pd.DataFrame([[location,sqft,bath,bhk]], columns=['location','total_sqft','bath','bhk'])
    
    prediction = pipe.predict(inputs)[0] * 100000 #for coverting to rupees
    print(prediction)

    return str(np.round(prediction,2))

if __name__ == "__main__":
    app.run(debug=True, port=5001)
