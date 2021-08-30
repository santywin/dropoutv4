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
import Ensembling as ensemble

def statistics(final,carrera, final_features):
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
    
    description = dataTrain.describe()
    
    dataTest = dataPred[dataPred.dropout == "2"]
    dataTest['dropout'].replace("2","",inplace = True)
   
    if final_features is False:
        dataTrain1 = dataTrain[["rateReprobadas","rateAnuladas","rateAprobadas","mediaAPRP","mediaAP","segMatActual","terMatActual"]]#,"terMatTot","numMatSem",
    else:
        dataTrain1 = dataTrain.loc[:,final_features.index.values]
        print("Selected features")
    x_train = dataTrain1.values[:,:]
    y_train = dataTrain['dropout'].astype(int).reset_index(drop = True)
   
    
    robust_scaler = preprocessing.StandardScaler()
    x_train = robust_scaler.fit_transform(x_train)
    print("Starting cross-validation (" +  str(len(x_train)) + ' learners)')
    
    seed = 7
    kfold = model_selection.KFold(n_splits=5, random_state=seed)
    model = RandomForestClassifier(n_estimators = 500)    
    
    test_size = 0.2
    X_train, X_test, Y_train, Y_test = model_selection.train_test_split(x_train, y_train, test_size=test_size, random_state=seed)
    #y_pred_bag, y_pred_bag_c = ensemble.bagging(X_train, X_test, Y_train, cfr, 10)
    y_pred_avrg = ensemble.averaging_predictions(X_train, X_test, Y_train)
    y_pred_calibrated = ensemble.model_calibration_prob(X_train, X_test, Y_train, model)
   
    
    scoring = 'accuracy'
    results = model_selection.cross_val_score(model, x_train, y_train, cv=kfold, scoring=scoring)
    print("Accuracy: %.3f (%.3f)" % (results.mean(), results.std()))
    accuracy = results.mean()
    print("Accuracy averaging")
    print(metrics.accuracy_score(Y_test, y_pred_avrg.round()))    
    print("Accuracy calibrated")
    print(metrics.accuracy_score(Y_test, y_pred_calibrated.round()))
    
    scoring = 'roc_auc'
    results = model_selection.cross_val_score(model, x_train, y_train, cv=kfold, scoring=scoring)
    print("AUC: %.3f (%.3f)" % (results.mean(), results.std()))
    auc_stat = results.mean()
    #metrics.roc_auc_score(Y_test, y_pred_bag)
    print("AUC averaging")
    print(metrics.roc_auc_score(Y_test, y_pred_avrg))    
    print("AUC calibrated")
    print(metrics.roc_auc_score(Y_test, y_pred_calibrated))

    scoring = 'neg_log_loss'
    results = model_selection.cross_val_score(model, x_train, y_train, cv=kfold, scoring=scoring)
    print("Logloss: %.3f (%.3f)" % (results.mean(), results.std()))
    log_loss = results.mean()
    print("Logloss averaging")
    print(metrics.log_loss(Y_test, y_pred_avrg))    
    print("Logloss calibrated")
    print(metrics.log_loss(Y_test, y_pred_calibrated))
    #metrics.log_loss(Y_test, y_pred_bag)
    
    
    X_train, X_test, Y_train, Y_test = model_selection.train_test_split(x_train, y_train, test_size=test_size, random_state=seed)
    model.fit(X_train, Y_train)
    predicted = model.predict(X_test)
    matrix = confusion_matrix(Y_test, predicted)
    print("Confusion matrix:")
    print(matrix)
    #metrics.confusion_matrix(Y_test, y_pred_bag_c)
    print("Confusion averaging")
    print(metrics.confusion_matrix(Y_test, y_pred_avrg.round()))    
    print("Confusion calibrated")
    print(metrics.confusion_matrix(Y_test, y_pred_calibrated.round()))
    
    report = classification_report(Y_test, predicted)
    print(report)
    print("Report averaging")
    print(classification_report(Y_test, y_pred_avrg.round()))    
    print("Report calibrated")
    print(classification_report(Y_test, y_pred_calibrated.round()))
    
    #report = classification_report(Y_test, y_pred_bag_c)
    #print(report)
    
    return accuracy, auc_stat, log_loss, description