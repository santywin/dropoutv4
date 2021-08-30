# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 20:09:21 2020

@author: Jon Imaz
"""

from sklearn.ensemble import IsolationForest
import numpy as np
data = final.copy()
dataPred = data[data.nombre_carrera == "BIOQUIMICA Y FARMACIA-REDISEÑO"]
    
#dataPred = data.drop("id",1).drop("degree",1)
#dataPred = data.drop("Unnamed: 0",1)
dataPred['dropout'].fillna("",inplace = True)
dataPred['dropout'] = dataPred['dropout'].astype(str)
dataPred['dropout'] = dataPred['dropout'].str.strip()    
dataPred['dropout'].fillna("2",inplace = True)
dataPred['dropout'].replace("","2",inplace = True)
dataPred = dataPred.fillna(0)
dataTrain = dataPred[dataPred.dropout != "2"]

dataTrain1 = dataTrain[["rateReprobadas","rateAnuladas","rateAprobadas","mediaAPRP","mediaAP","segMatActual","terMatActual","terMatTot"]]#"numMatSem",
    
clf = IsolationForest(  max_samples=100, random_state = 1, contamination= 'auto')
preds = clf.fit_predict(dataTrain1)
print(preds)
list(preds).count(-1)

from sklearn.cluster import DBSCAN
outlier_detection = DBSCAN(min_samples = 2, eps = 3)
clusters = outlier_detection.fit_predict(dataTrain1)
list(clusters).count(-1)
