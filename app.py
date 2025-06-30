from flask import Flask, request,jsonify,render_template,app
import pandas as pd
import numpy as np
import pickle
#from flask import Response

app = Flask(__name__)


model = pickle.load(open("model/model.pkl",'rb'))
scaler = pickle.load(open("model/Scaler.pkl",'rb'))

@app.route('/',methods = ['GET','POST'])
def predict_datapoint():
    #return render_template('index.html')

#@app.route('/diseaseprediction',methods=['GET', 'POST'])
#def predict_datapoint():
    result=""

    if request.method == 'POST':
        Pregnancies = int(request.form.get("Pregnancies"))
        Glucose = float(request.form.get('Glucose'))
        BloodPressure = float(request.form.get('BloodPressure'))
        SkinThickness = float(request.form.get('SkinThickness'))
        Insulin = float(request.form.get('Insulin'))
        BMI = float(request.form.get('BMI'))
        DiabetesPedigreeFunction = float(request.form.get('DiabetesPedigreeFunction'))
        Age = float(request.form.get('Age'))

        new_data = scaler.transform([[Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,Age]])
        predict = model.predict(new_data)

        if predict[0] == 1:
            result = 'Diabetic'
        else:
            result = 'Non-Diabetic'

        return render_template('single_prediction.html',result=result)
    else:
        return render_template('index.html')



if __name__=='__main__':
    app.run(debug=True)
