from flask import Flask,render_template,request,redirect
from flask_cors import CORS,cross_origin
import pickle
import pandas as pd
import numpy as np

app=Flask(__name__)
bike=CORS(app)
model=pickle.load(open('LinearRegressionModel.pkl','rb'))
bike=pd.read_csv('new_used.csv')

@app.route('/',methods=['GET','POST'])
def index():
    companies=sorted(bike['brand'].unique())
    bike_models=sorted(bike['bike_name'].unique())
    year=sorted(bike['age'].unique(),reverse=True)
    cities=bike['city'].unique()
    companies=np.insert(companies,0,'Select Company')
    year=np.insert(year,0,0)
    bike_models=np.insert(bike_models,0,'Select Model')
    cities=np.insert(cities,0,'Select City')

    
   
    return render_template('index.html',companies=companies, bike_models=bike_models, years=year,cities=cities)


@app.route('/predict',methods=['POST'])
@cross_origin()
def predict():

    company=request.form.get('company')
    car_model=request.form.get('car_models')
    year=int(float(request.form.get('year')))
    fuel_type=request.form.get('fuel_type')
    driven=int(float(request.form.get('kilo_driven')))
    print(car_model,company,year)

    prediction=model.predict(pd.DataFrame([[car_model,company,year,driven,fuel_type]],columns=['bike_name','brand','age','kms_driven','city']))
    

    return str(np.round(prediction[0],2))



if __name__=='__main__':
    app.run(debug=True)