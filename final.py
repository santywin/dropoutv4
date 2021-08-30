import pandas as pd 

def finalData(dataMedia, cambioEstudiante, credPass,dropout):    
   
    dataMedia = dataMedia.copy().astype(object)
    dataEstudiante = cambioEstudiante.copy().astype(object)    
    dataCredCurs = credPass.copy().astype(object)
    
    dataEstudiante["codigo_carrera"] = dataEstudiante["codigo_carrera"].astype(int)
    dataMedia["codigo_carrera"] = dataMedia["codigo_carrera"].astype(int)
    dataCredCurs["codigo_carrera"] = dataCredCurs["codigo_carrera"].astype(int)
    
    data = pd.merge(dataEstudiante, dataMedia, how='inner', on=['id','codigo_carrera'])
    data = pd.merge(data, dataCredCurs, how='inner', on=['id','codigo_carrera'])
    
    dataTot = data.groupby(['nombre_carrera'],as_index=False).id.count()
    dataTot.columns = ['nombre_carrera','TOT']
        
    #Ponemos el dropout de los 2 años sin acudir a clase    
    dropout = dropout.astype(object)
    dropout["codigo_carrera"] = dropout["codigo_carrera"].astype(int)
    
    data = pd.merge(data, dropout, how='inner', on=['id','codigo_carrera'])
    data['porcentaje_carrera'] = data['numero_horas_aprobados']/data['total_horas_carrera'].astype(float)
    for i in range(len(data)):
        if data.ultMatricula.values[i] <= 2:
            data.dropout.values[i] = 2
        if data.porcentaje_carrera.values[i] >= 0.9 :
            data.dropout.values[i] = 1
    
    data.loc[data['graduado'] == "1",'dropout'] = 1
    data.drop("graduado",1,inplace = True)
    dataPass = data[data.dropout == 1]
    dataPass = dataPass.groupby(['nombre_carrera'],as_index=False).id.count()
    dataPass.columns = ['nombre_carrera','PASS']
    dataFail =  data[data.dropout == 0]
    dataFail = dataFail.groupby(['nombre_carrera'],as_index=False).id.count()
    dataFail.columns = ['nombre_carrera','FAIL']
    
    dataRate = pd.merge(dataTot, dataPass, how='outer', on=['nombre_carrera'])
    dataRate = pd.merge(dataRate, dataFail, how='outer', on=['nombre_carrera'])        
    dataRate.fillna(0,inplace = True)
        
    #No se tiene en cuenta el número de estudiantes que están cursando ahora para el total de los alumnos
    dataRate['rate'] = dataRate.FAIL/(dataRate.FAIL+dataRate.PASS)
    dataRate = dataRate.drop("TOT",1).drop("PASS",1).drop("FAIL",1).fillna(0)
    data = data.astype(object)
    dataRate = dataRate.astype(object)
    data = pd.merge(data, dataRate, how='outer', on=['nombre_carrera'])
    data = data.drop_duplicates()        
    
    data = data.sort_values(by='id', ascending=True)
    data['codigo_carrera'] = data['codigo_carrera'].fillna(0)
    data = data[data['codigo_carrera']!= 0]
    
    return data