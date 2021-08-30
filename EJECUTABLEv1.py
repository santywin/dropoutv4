import downDB
import get_estudiantes
import get_calificaciones
import calcVar
import final as finalpy
#import predRF
import pandas as pd
import predict
import predictEnsemble
import stats_pred
from sqlalchemy import create_engine
#import psycopg2, psycopg2.extras
from numpy import inf
import numpy as np
import features
import os      
import warnings
warnings.filterwarnings("ignore")
from sklearn.metrics import accuracy_score
from datetime import date, time, datetime
#Cambiar para marcar el semestre de graduados
SEMESTRE_GRADUADOS = 7

#file = 'estudiantes.csv'
#estudiantes = pd.read_csv(file)
#file = 'calificaciones.csv'
#calificaciones = pd.read_csv(file, encoding = "ISO-8859-1")

print("Prediciendo Dropout...\n")
#descargar datos para estudiantes y calificaciones de la BD 
estudiantes,calificaciones = downDB.download()




estudiantes = estudiantes.astype({'id': 'int','codigo_carrera': 'int', 'codigo_malla': 'int', 'total_horas_carrera': 'int', 'total_asignaturas': 'int', 'minnivel': 'int', 'edad': 'int', 'quintil': 'int'})

estudiantes['fecha_nacimiento'] = pd.to_datetime(estudiantes["fecha_nacimiento"], format = '%Y-%m-%d')


estudiantes["estudia_casa"] = estudiantes.apply(lambda x: "SI" if str(x["ciudaddomicilio"]) == str(x["sede"]) else "NO", axis=1)


estudiantes.info()


    
calificaciones['id'] = calificaciones['id'].astype(int)
calificaciones['codigo_curriculum'] = calificaciones['codigo_curriculum'].astype(int)
calificaciones['codigo_asignatura'] = calificaciones['codigo_asignatura'].astype(int)
calificaciones['nota_final'] = calificaciones['nota_final'].astype(int)
calificaciones['numero_matricula'] = calificaciones['numero_matricula'].astype(int)
calificaciones['codigo_carrera'] = calificaciones['codigo_carrera'].astype(int)
calificaciones['total_horas_asignatura'] = calificaciones['total_horas_asignatura'].astype(int)
calificaciones['anio_curso_asignatura'] = calificaciones['anio_curso_asignatura'].astype(int)
calificaciones['semestre_en_carrera'] = calificaciones['semestre_en_carrera'].astype(int)
calificaciones['semestre_cursa'] = calificaciones['semestre_cursa'].astype(int)
calificaciones['numero_creditos'] = calificaciones['numero_creditos'].astype(int)

calificaciones.info()

calificaciones.loc[calificaciones['semestre_en_carrera']>calificaciones['semestre_cursa'],'semestre_cursa'] = calificaciones['semestre_en_carrera']

#Etiquetamos a los graduados
estudiantes.loc[estudiantes["minnivel"] >= SEMESTRE_GRADUADOS, "graduado"] = "S"
estudiantes.drop("minnivel",1,inplace = True)

print("Prediciendo Dropout...\n")
#estudiantes,calificaciones = downDB.download()
print("Archivos descargados de BD\n")
print("Número total de estudiantes: " + str(estudiantes.id.count()))
print("Número total de estudiantes en el historial: "+ str(calificaciones.drop_duplicates(subset = 'id').id.count()))
cambioEstudiante = get_estudiantes.cambioEst(estudiantes)
cambioHistorial = get_calificaciones.cambCal(calificaciones)
cambioHistorial = cambioHistorial.sort_values(["id","codigo_carrera","semestre_cursa"])
#Asignaturas reprobadas con 0 no se toman en cuenta
cambioHistorial = cambioHistorial[cambioHistorial['nota_final']!=0]

#NÃºmero de semestres que lleva cursando cada alumno
asigxSem = cambioHistorial.groupby(["id","codigo_carrera","semestre_cursa"],as_index = False).codigo_asignatura.count()
alumSem = asigxSem.groupby(["id","codigo_carrera"],as_index = False).count().drop("codigo_asignatura",1)
alumSem.columns = ["id","codigo_carrera","Semestres"]

#Cogemos los estudiantes del primer semestre
primerSem = cambioHistorial.groupby(["id","codigo_carrera"], as_index = False).semestre_cursa.min().astype(object)
cambioHistorial1Sem = cambioHistorial.merge(primerSem, on=['id','codigo_carrera','semestre_cursa'], how='inner')
alum1Sem = alumSem[alumSem['Semestres'] == 1]
al1Sem = alumSem[alumSem['Semestres'] >= 1]
#Cogemos el primer y segundo semestre
cambioHistorial2 = pd.concat([cambioHistorial, cambioHistorial1Sem]).drop_duplicates(keep=False)
segSem = cambioHistorial2.groupby(["id","codigo_carrera"], as_index = False).semestre_cursa.min().astype(object)
cambioHistorial2Sem = cambioHistorial.merge(segSem, on=['id','codigo_carrera','semestre_cursa'], how='inner')
cambioHistorial12Sem = pd.concat([cambioHistorial1Sem, cambioHistorial2Sem])
alum2Sem = alumSem[alumSem['Semestres'] == 2]
al2Sem = alumSem[alumSem['Semestres'] >= 2]
#Primer, segundo y tercer semestre
cambioHistorial3 = pd.concat([cambioHistorial, cambioHistorial12Sem]).drop_duplicates(keep=False)
terSem = cambioHistorial3.groupby(["id","codigo_carrera"], as_index = False).semestre_cursa.min().astype(object)
cambioHistorial3Sem = cambioHistorial.merge(terSem, on=['id','codigo_carrera','semestre_cursa'], how='inner')
cambioHistorial123Sem = pd.concat([cambioHistorial12Sem, cambioHistorial3Sem])
alum3Sem = alumSem[alumSem['Semestres'] == 3]
al3Sem = alumSem[alumSem['Semestres'] >= 3]
#Primer, segundo, tercer y cuarto semestre
cambioHistorial4 = pd.concat([cambioHistorial, cambioHistorial123Sem]).drop_duplicates(keep=False)
cuatSem = cambioHistorial4.groupby(["id","codigo_carrera"], as_index = False).semestre_cursa.min().astype(object)
cambioHistorial4Sem = cambioHistorial.merge(cuatSem, on=['id','codigo_carrera','semestre_cursa'], how='inner')
cambioHistorial1234Sem = pd.concat([cambioHistorial123Sem, cambioHistorial4Sem])
alum4Sem = alumSem[alumSem['Semestres'] == 4]
al4Sem = alumSem[alumSem['Semestres'] >= 4]
#Primer, segundo, tercer, cuarto y quinto semestre
cambioHistorial5 = pd.concat([cambioHistorial, cambioHistorial1234Sem]).drop_duplicates(keep=False)
quinSem = cambioHistorial5.groupby(["id","codigo_carrera"], as_index = False).semestre_cursa.min().astype(object)
cambioHistorial5Sem = cambioHistorial.merge(quinSem, on=['id','codigo_carrera','semestre_cursa'], how='inner')
cambioHistorial12345Sem = pd.concat([cambioHistorial1234Sem, cambioHistorial5Sem])
alum5Sem = alumSem[alumSem['Semestres'] == 5]
al5Sem = alumSem[alumSem['Semestres'] >= 5]
#Alumnos a partir de 5 semestre
alumMas5Sem = alumSem[alumSem['Semestres'] > 5]


resultados = pd.DataFrame()
datosBD = pd.DataFrame()
carrerasSinActivos = ""
carrerasSinActivosAUC = ""
abandonoXSem = pd.DataFrame()
abandonoXSem['carrera'] = cambioEstudiante['nombre_carrera'].drop_duplicates().dropna()
abandonoXSem.set_index('carrera',inplace = True)
gradXSem = pd.DataFrame()
gradXSem['carrera'] = cambioEstudiante['nombre_carrera'].drop_duplicates().dropna()
gradXSem.set_index('carrera',inplace = True)
correlation = {}
featureSelectionRF = {}
featureSelection = {}
stats = pd.DataFrame(index='Accuracy AUC Log_loss'.split())

carreras = pd.DataFrame(estudiantes['nombre_carrera'].drop_duplicates().dropna())
try:
    os.mkdir('Archivos')
except Exception as e:
    print("Error 1 ", e)
    pass
for carrera in carreras['nombre_carrera']:
    try:
        os.mkdir('Archivos/' + carrera)
    except Exception as e:
        print("Error 2 ", e)
        pass

for i in range(6):
    if i == 0:
        print("Semestre 1")
        cambioHistorial1Sem = cambioHistorial1Sem.copy()
        cambioHistorialSemActual = cambioHistorial1Sem.copy()
        alum1Sem = alum1Sem.copy()
        al1Sem = al1Sem.copy()
    elif i == 1:
        print("Semestre 2")
        cambioHistorial1Sem = cambioHistorial12Sem.copy()
        cambioHistorialSemActual = cambioHistorial2Sem.copy()
        alum1Sem = alum2Sem.copy()
        al1Sem = al2Sem.copy()
    elif i == 2:
        print("Semestre 3")
        cambioHistorial1Sem = cambioHistorial123Sem.copy()
        cambioHistorialSemActual = cambioHistorial3Sem.copy()
        alum1Sem = alum3Sem.copy()
        al1Sem = al3Sem.copy()
    elif i == 3:
        print("Semestre 4")
        cambioHistorial1Sem = cambioHistorial1234Sem.copy()
        cambioHistorialSemActual = cambioHistorial4Sem.copy()
        alum1Sem = alum4Sem.copy()
        al1Sem = al4Sem.copy()
    elif i == 4:
        print("Semestre 5")
        cambioHistorial1Sem = cambioHistorial12345Sem.copy()
        cambioHistorialSemActual = cambioHistorial5Sem.copy()
        alum1Sem = alum5Sem.copy()
        al1Sem = al5Sem.copy()
    elif i == 5:
        print("Semestre 6")
        cambioHistorial1Sem = cambioHistorial.copy()
        cambioHistorialSemActual = cambioHistorial5Sem.copy()
        alum1Sem = alumMas5Sem.copy()
        al1Sem = al5Sem.copy()
    dropout, dataMedia, credPass = calcVar.calcular(cambioHistorial,cambioEstudiante,cambioHistorial1Sem,cambioHistorialSemActual)
    dropout = dropout.reset_index(level=['id', 'codigo_carrera'])
    print("Variables de entrada calculadas\n")   
    final = finalpy.finalData(dataMedia, cambioEstudiante, credPass,dropout)
    final = final.drop_duplicates(subset = "id")
    print("Datos finales listos\n")
    carreras = pd.DataFrame(final['nombre_carrera'])
    carreras = carreras.drop_duplicates().dropna()
    abandonoXSem['dropoutSem'+str(i+1)] = 0
    gradXSem['gradSem'+str(i+1)] = 0                                    
    final.dropout = final.dropout.astype(str)
    #Se han observado muchos alumnos que han abandonado pero no por temas academicos
    dropoutNoAcad = final[((final['rateAprobadas']>=0.95) & (final['dropout'] == "0"))]    
    dropoutNoAcad =  dropoutNoAcad.merge(alum1Sem, on=['id','codigo_carrera'], how='inner')
    datosBD = pd.concat([datosBD, dropoutNoAcad],axis = 0)  
    final = final[~((final['rateAprobadas']>=0.95) & (final['dropout'] == "0"))]
    #Quitamos los alumnos que hayan abandonado ya (mejor para la predicción final puesto que predice valores más bajos 
    #de abandono pero peor para las estadísticas)
    """
    final1 = final[final['dropout']== "0"]
    final1 = final1.merge(al1Sem, on=['id','codigo_carrera'], how='inner')
    final2 = final[final['dropout']!= "0"]
    final = pd.concat([final1,final2],ignore_index = True).drop("Semestres",1)
    """    
    #final.loc[final['porcentaje_carrera']>=np.mean(final['porcentaje_carrera']*2),'dropout'] = "0"
    for carrera in carreras['nombre_carrera']:
        print("\n#####################################################################\n")
        print("\n"+carrera + str(i+1) +"\n")
        try:
            os.mkdir('Archivos/' + carrera + '/' + carrera + str(i+1))
        except Exception as e:
            print("Error 3 ", e)
            pass

        #carrera = "INGENIERÍA CIVIL"

        try:
            final = final.fillna(0)            
            final = final.replace([np.inf, -np.inf], 0)
            featureSelection[carrera + "" +str(i+1)] = features.selection(final, carrera, i)
            accuracy, auc_stat, log_loss, description = stats_pred.statistics(final, carrera)
            description.to_excel('Archivos/' + carrera + '/' + carrera + str(i+1) + '/' + "description.xlsx")
            stats[carrera + "" +str(i+1)] = 0.1
            stats[carrera + "" +str(i+1)]["Accuracy"] = accuracy
            stats[carrera + "" +str(i+1)]["AUC"] = auc_stat
            stats[carrera + "" +str(i+1)]["Log_loss"] = log_loss
        except Exception as e:
            print("Error 4 ", e)
            stats[carrera + "" +str(i+1)] = 0.0   
            print("Error")
        try:
            #featureSelectionRF[carrera + "" +str(i+1)] = features.selection(finalTot, carrera)
            #finalData1 = predict.predict(final, carrera)
            finalData1 = predictEnsemble.predict(final, carrera, i)
            finalData1 =  finalData1.merge(alum1Sem, on=['id','codigo_carrera'], how='inner')
            abandSemX = finalData1[finalData1['dropout'] == 0]            
            gradSemX = finalData1[finalData1['dropout'] == 1]             
            abandonoXSem['dropoutSem'+str(i+1)][carrera] = abandSemX['id'].count()
            gradXSem['gradSem'+str(i+1)][carrera] = gradSemX['id'].count()
            finalData1['Accuracy'] = accuracy
            datosBD = pd.concat([datosBD, finalData1],axis = 0)
        except Exception as e:
            print("Error 5 ", e)
            print("Error predict")   
            carrerasSinActivos = carrerasSinActivos  + carrera+ str(i+1)+ ","
            final1 = final[final["nombre_carrera"] == carrera]
            final1["codigo_carrera"] =  final1["codigo_carrera"].astype(int)
            final1 =  final1.merge(alum1Sem, on=['id','codigo_carrera'], how='inner')
            final1['Accuracy'] = 0.1
            datosBD = pd.concat([datosBD, final1],axis = 0, sort = True)
                             
datosBD["dropout"] = datosBD["dropout"].replace("3","0").astype(str)

datosBD.rename(columns={'rate':'dropoutCarrera'}, inplace=True)
datosBD.rename(columns={'dropout':'dropout_RF'}, inplace=True)
datosBD.rename(columns={'dropout_avrg':'dropout'}, inplace=True)
datosBD.drop_duplicates(subset='id', inplace =True)

datosBD["dropout_RF"] = datosBD["dropout_RF"].fillna(-999).astype(str)

##Subimos resultados a la base de datos
print("Subiendo probabilidad de abandono")

abandonoXSem['carreras'] = abandonoXSem.index
engine = create_engine('postgresql://lala:PB2Cx3fDEgfFTpPn@172.16.101.55:5432/lala')
datosBD.to_sql('dropoutporsemestres', engine,if_exists = 'replace', index=False)
abandonoXSem.to_sql('abandono', engine,if_exists = 'replace', index=False)

print("Predicción acabada.")
try:
    os.mkdir('Resultados/')
except Exception as e:
    print("Error 5 ", e)
    print("Error predict") 
    pass
stats.to_excel("Resultados/metricas.xlsx")
datosBD.to_excel("Resultados/predicciones.xlsx")
abandonoXSem.reset_index(level=0, inplace=True)
abandonoXSem.to_excel("Resultados/dropoutNumberPerSemesterAndDegree.xlsx")
gradXSem.reset_index(level=0, inplace=True)
gradXSem.to_excel("Resultados/graduateNumberPerSemesterAndDegree.xlsx")
