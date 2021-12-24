# -*- coding: utf-8 -*-
"""
Created on Fri Dec 24 14:19:51 2021

@author: hp
"""
from flask import Flask,request
import numpy as np 
import joblib
from flasgger import Swagger
from pickle import load

app = Flask(__name__)
Swagger(app)

loaded_model = load(open('best_model.pkl', 'rb'))
poly = load(open('poly.pkl', 'rb'))
sc = load(open('scaler.pkl', 'rb'))

@app.route('/',methods = ['Get'])
def predict():
    
    l = []
    i1 = request.args.get('House Age')
    print('1')
    l.append(i1)
    
    i2 = request.args.get('Distance To The Nearest MRT Station')
    print('2')
    l.append(i2)
    
    i3 = request.args.get('number of convenience store')
    print('3')
    l.append(i3)
    
    i4 = request.args.get('latitude')
    print('4')
    l.append(i4)
    
    i5 = request.args.get('longitude')
    print('5')
    l.append(i5)
    
    
    arr = np.asarray([l])
    
    arr = poly.transform(arr)
    
    scaler_arr = sc.transform(arr)
    
    p = round(loaded_model.predict(scaler_arr)[0][0],2)
    
    return 'Price of the house per unit area:'+str(p)
    


if __name__=='__main__':
    app.run()
    
    
    
    
    
    
    
    
    
    
    