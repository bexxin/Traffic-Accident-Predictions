# -*- coding: utf-8 -*-
"""
Created on Thu Feb 15 10:59:35 2024

@author: becky
"""
#Removed date fields

from flask import Flask, request, render_template
import joblib
import pandas as pd
import numpy as np
from datetime import datetime


app=Flask(__name__)

#load pipelines
#model=joblib.load("./final_pipeline.pkl")
lr_pipeline=joblib.load("./Logistic Regression_pipeline.pkl")
dt_pipeline=joblib.load("./Decision Tree_pipeline.pkl")
svm_pipeline=joblib.load("./SVM_pipeline.pkl")
rf_pipeline=joblib.load("./Random Forest_pipeline.pkl")
nn_pipeline=joblib.load("./Neural Network_pipeline.pkl")


#route the app
@app.route("/")
#associate route with a function-render templates(html files in the templates folder)
def home():
    return render_template("index.html")

@app.route("/result", methods=["POST"])

def result():
   #get values from form
    #date_input=request.form.get("dateInput")
    time_input=request.form.get("timeInput")
    ROAD_CLASS=request.form.get("roadClass")
    LATITUDE=float(request.form.get("latitudeInput"))
    LONGITUDE=float(request.form.get("longitudeInput"))
    LOCCOORD=request.form.get("locCoord")
    TRAFFCTL=request.form.get("traffctl")
    VISIBILITY=request.form.get("visibiility")
    LIGHT=request.form.get("light")
    RDSFCOND=request.form.get("surface-condition")
    PEDESTRIAN_Yes=request.form.get("pedestrian")
    CYCLIST_Yes=request.form.get("cyclist")
    AUTOMOBILE_Yes=request.form.get("automobile")
    MOTORCYCLE_Yes=request.form.get("motorcycle")
    TRUCK_Yes=request.form.get("truck")
    TRSN_CITY_VEH_Yes=request.form.get("trsn_city_veh")
    EMERG_VEH_Yes=request.form.get("emerg_veh")
    SPEEDING_Yes=request.form.get("speeding")
    AG_DRIV_Yes=request.form.get("ag_drive")
    REDLIGHT_Yes=request.form.get("red_light")
    ALCOHOL_Yes=request.form.get("alcohol")
    DISABILITY_Yes=request.form.get("disability")

    classifier=request.form.get("classifier")
    
    
    time_obj=datetime.strptime(time_input, "%H:%M")
    TIME=time_obj.strftime("%H%M")
    TIME=TIME.zfill(4)
    print(TIME)

    columns=["TIME", "ROAD_CLASS", "LATITUDE", "LONGITUDE","LOCCOORD","TRAFFCTL","VISIBILITY","LIGHT","RDSFCOND",
             "PEDESTRIAN_Yes","CYCLIST_Yes","AUTOMOBILE_Yes","MOTORCYCLE_Yes","TRUCK_Yes","TRSN_CITY_VEH_Yes","EMERG_VEH_Yes",
             "SPEEDING_Yes","AG_DRIV_Yes","REDLIGHT_Yes","ALCOHOL_Yes","DISABILITY_Yes"]
    features=pd.DataFrame([[TIME,ROAD_CLASS,LATITUDE,LONGITUDE,LOCCOORD,TRAFFCTL,VISIBILITY,LIGHT,RDSFCOND,
             PEDESTRIAN_Yes,CYCLIST_Yes,AUTOMOBILE_Yes,MOTORCYCLE_Yes,TRUCK_Yes,TRSN_CITY_VEH_Yes,EMERG_VEH_Yes,
             SPEEDING_Yes,AG_DRIV_Yes,REDLIGHT_Yes,ALCOHOL_Yes,DISABILITY_Yes]],columns=columns)
    

    prediction=predict(classifier, features)
    
    return render_template("result.html",prediction=prediction)

def predict(classifier,features):
    if classifier == 'lr':
        model=lr_pipeline
    elif classifier =='dt':
        model=dt_pipeline
    elif classifier =='svm':
        model=svm_pipeline
    elif classifier == 'rf':
        model=rf_pipeline
    elif classifier == 'nn':
        model=nn_pipeline
        
    prediction=model.predict(features)
    return prediction[0]

#run the app
if __name__=="__main__":
    app.run(debug=True, port=5000)