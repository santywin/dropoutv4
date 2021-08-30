# -*- coding: utf-8 -*-
"""
Created on Wed Mar  4 15:46:52 2020

@author: Jon Imaz
"""
import numpy as np
import pandas as pd 
from sklearn import model_selection
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

def statistics(final,carrera):
    try:
        data = final.copy()
        dataPred = data[data.nombre_carrera == carrera]
       #dataPred = data[data.nombre_carrera == "INGENIERIA CIVIL-REDISEÑO"]
        dataPred = dataPred.drop("id",1)
        
            
        dataPred['dropout'].fillna("2",inplace = True)
        #print(dataPred.stateStudent)
        dataPred = dataPred.dropna()
        #print(len(dataPred))    
        dataPred['dropout'].replace("","2",inplace = True)
        dataPred = dataPred.fillna(0)
        
        dataTrain = dataPred[dataPred.dropout != "2"]
        
        mediaCarrera = dataTrain.groupby(['nombre_carrera']).mediaAP.sum().to_frame()/dataTrain.groupby(['nombre_carrera']).mediaAP.count().to_frame()
        mediaCarreraRateP = dataTrain.groupby(['nombre_carrera']).rateAprobadas.sum().to_frame()/dataTrain.groupby(['nombre_carrera']).rateAprobadas.count().to_frame()
        
        aprobados = dataTrain[dataTrain['dropout'] == "1"]
        abandonados = dataTrain[dataTrain['dropout'] == "0"]
        abandonados = abandonados[abandonados['mediaAP'] != 0]
           
        for i in range(len(dataTrain)):
            if  dataTrain['rateAprobadas'].values[i] >= mediaCarreraRateP['rateAprobadas'].values[0]*1.05 and dataTrain['dropout'].values[i] == "0" and dataTrain['mediaAP'].values[i] >= mediaCarrera['mediaAP'].values[0]*1.05:
                dataTrain['dropout'].values[i] = "3"
        
        dataTrain = dataTrain[dataTrain['dropout']!= "3"]
        dataTrain.dropout = dataTrain.dropout.astype(int)
        dataTrain = dataTrain.drop("nombre_carrera",1)
        
        dataTest = dataPred[dataPred.dropout == "2"]
        dataTest['dropout'].replace("2","",inplace = True)
        description = dataTrain.describe()
        
        dataTrain1 = dataTrain[["rateReprobadas","rateAprobadas","mediaAPRP","mediaAP","segMatActual","terMatActual","terMatTot"]]#"numMatSem",
        x_train = dataTrain1.values[:,:]
        y_train = dataTrain['dropout'].astype(int).reset_index(drop = True)
        
        #x_test = dataTest.values[:,[3,5,6,7,8,9]]
        #y_test = dataTest.values[:,13]
        
        robust_scaler = preprocessing.StandardScaler()
        x_train = robust_scaler.fit_transform(x_train)
        #x_test = robust_scaler.fit_transform(x_test)
        print("Starting cross-validation (" +  str(len(x_train)) + ' learners)')
        
        seed = 7
        kfold = model_selection.KFold(n_splits=5, random_state=seed)
        model = RandomForestClassifier(n_estimators = 500)
        scoring = 'accuracy'
        results = model_selection.cross_val_score(model, x_train, y_train, cv=kfold, scoring=scoring)
        print("Accuracy: %.3f (%.3f)" % (results.mean(), results.std()))
        accuracy = results.mean()
        
        #print(y_train)
        #print(np.sum(y_train == 0))
        #print(np.sum(y_train == 1))
        
        from sklearn.utils import shuffle
        X_s, y_s = shuffle(x_train, y_train)

        
        scoring = 'roc_auc'
        results = model_selection.cross_val_score(model, X_s, y_s, cv=kfold, scoring=scoring)
        print("AUC: %.3f (%.3f)" % (results.mean(), results.std()))
        auc_stat = results.mean()
        
        scoring = 'neg_log_loss'
        results = model_selection.cross_val_score(model, X_s, y_s, cv=kfold, scoring=scoring)
        print("Logloss: %.3f (%.3f)" % (results.mean(), results.std()))
        log_loss = results.mean()
        
        test_size = 0.33
        X_train, X_test, Y_train, Y_test = model_selection.train_test_split(x_train, y_train, test_size=test_size, random_state=seed)
        model.fit(X_train, Y_train)
        predicted = model.predict(X_test)
        matrix = confusion_matrix(Y_test, predicted)
        print("Confusion matrix:")
        print(matrix)
        
        report = classification_report(Y_test, predicted)
        print(report)
        
    except Exception as e:
            print("Error 7 ", e)
    
    return accuracy, auc_stat, log_loss, description
