# -*- coding: utf-8 -*-
"""
Created on Fri Feb  7 19:17:59 2020

@author: Jon Imaz
"""
import pandas as pd
import numpy as np
import psycopg2, psycopg2.extras

file = 'datosBD.csv'
data = pd.read_csv(file)
data.drop_duplicates(subset='id', inplace =True)
filesin = 'datosBDsinprimsem.csv'
datasin = pd.read_csv(file)
filesinsin = 'datosBDsindropoutsinprimsem.csv'
datasinsin = pd.read_csv(file)


###########BASE DE DATOS#####################
conn = psycopg2.connect(database='lala_ucuenca',user='lala',password='postgres', host='10.0.2.60')

cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)

cur.execute("SELECT * FROM dropoutporsemestres")# ORDER BY id")
dataBD =cur.fetchall();
dfBD = pd.DataFrame(np.array(dataBD), columns = dataBD[0].keys())
dfBD.to_excel("datosCuencaDropout.xlsx")
#dfBD = dfBD[dfBD['codigo_carrera']==7]
