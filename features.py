# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 12:57:03 2019

@author: Jon Imaz
"""

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import GridSearchCV
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

def selection(final,carrera, i):
    
    #file = 'Final.csv'
    #data = pd.read_csv(file)
    data = final.copy()
    dataPred = data[data.nombre_carrera == carrera]
        
    #dataPred = data.drop("id",1).drop("degree",1)
    #dataPred = data.drop("Unnamed: 0",1)
    dataPred['dropout'].fillna("",inplace = True)
    dataPred['dropout'] = dataPred['dropout'].astype(str)
    dataPred['dropout'] = dataPred['dropout'].str.strip()    
    dataPred['dropout'].fillna("2",inplace = True)
    dataPred['dropout'].replace("","2",inplace = True)
    dataPred = dataPred.fillna(0)
    dataTrain = dataPred[dataPred.dropout != "2"]
    #Eliminamos estudiantes que meten ruido
    
    dataTest = dataPred[dataPred.dropout == "2"]
    dataTest['dropout'].replace("2","",inplace = True)
    dataTest = dataTest.drop("id",1).drop("nombre_carrera",1)
   
    dataTrain1 = dataTrain[["rateReprobadas","rateAprobadas","mediaRP","mediaAPRP","mediaAP","segMatActual","segMatTot","terMatActual","terMatTot","rateAprobadasActual","mediaAPRPActual"]]#"numMatSem",
    
    x_train, x_test, y_train, y_test = train_test_split(dataTrain1, dataTrain[['dropout']], test_size=0.3, random_state=0)
    
    
    cfr = RandomForestClassifier(n_estimators = 100)
    cfr.fit(x_train,y_train)
    
    sel = SelectFromModel(cfr,threshold=0.1)
    sel.fit(x_train, y_train)
    names = sel.get_support()
    selected_feat = x_train.columns[(sel.get_support())]
    len(selected_feat)    
    print(selected_feat)
    importances = sel.estimator_.feature_importances_
    indices = np.argsort(importances)[::-1] 
    plt.figure()
    plt.title("Feature importances")
    plt.bar(range(dataTrain1.shape[1]), importances[indices], color="r", align="center")
    #plt.xticks(range(dataTrain1.shape[1]), indices)
    plt.xticks(range(dataTrain1.shape[1]), x_train.columns[indices], rotation = 90)
    plt.xlim([-1, dataTrain1.shape[1]])
    #plt.show() 
    plt.savefig('Archivos/' + carrera + '/' + carrera + str(i+1) + '/' +'importances.png', bbox_inches='tight')
    
    
    X_important_train = sel.transform(x_train)
    X_important_test = sel.transform(x_test)
    
    clf_important = RandomForestClassifier(n_estimators=100)
    clf_important.fit(X_important_train, y_train)    

    y_pred = cfr.predict(x_test)
    accuracy_score(y_test, y_pred)        
    
    
    y_important_pred = clf_important.predict(X_important_test)    
    accuracy_score(y_test, y_important_pred)
    
    
    selected = dataTrain1.columns[names].to_frame()
    
    
    return selected
