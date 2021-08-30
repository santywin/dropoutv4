import pandas as pd 
import time
import numpy as np

def calcular(cambioHistorial,cambioEstudiante,cambioHistorial1Sem,cambioHistorialSemActual):
    #Datos desde que empezó el estudiante hasta el semestre actual
    dataHastaActual = cambioHistorial1Sem.copy()
    #datos tomados del semestre actual
    dataActual = cambioHistorialSemActual.copy()
    #Dropout despues de 2 años sin matricularse y vuelve a cursar otra carrera
    cambioHistorial['dropout'] = 2
    #Datos completos del estudiante
    dataHistCompleto = cambioHistorial.sort_values(['id','codigo_carrera', 'anio_curso_asignatura'], ascending=[True, True,True])
    
    #Si queremos centrarnos en un solo estudiante 
    #dataHistCompleto = dataHistCompleto[dataHistCompleto['id']== 28652]
    #dataHastaActual = dataHastaActual[dataHastaActual['id']== 28652]
    #dataActual = dataActual[dataActual['id']== 28652]
    
    years = dataHistCompleto['anio_curso_asignatura']
    ids = dataHistCompleto['id']
    carrera = dataHistCompleto['codigo_carrera']
    for i in range(len(years)):
    	if ids.values[i] == ids.values[i-1] and carrera.values[i] != carrera.values[i-1]:
    		if (years.values[i]-years.values[i-1])>2:
    			dataHistCompleto['dropout'].values[i] = 0
    		else:
    			dataHistCompleto['dropout'].values[i] = 1
    	else:
    		dataHistCompleto['dropout'].values[i] = 1
    
    dropid =pd.concat([ids,carrera, dataHistCompleto['dropout']], axis=1,)	 
    dropid.columns = ['id','codigo_carrera','dropout']
    dropout = dropid.groupby(['id','codigo_carrera']).dropout.sum().to_frame()
    dropout.columns = ['dropout']
    ids = dropout.index.values
    ids = pd.DataFrame(ids)
    
    #Si lleva mas de los ultimos 2 años sin matricularse y no se ha graduado
    datamax = dataHistCompleto.groupby(['id','codigo_carrera']).anio_curso_asignatura.max().to_frame()
    datamax.columns = ['year']	
    yearmax = datamax['year']
    añoActual = int(time.strftime("%Y"))    
   
    for i in range(len(yearmax)):
    	if añoActual >= (yearmax.values[i]+2): #and stateStud.values[i] != 0:
    		dropout['dropout'].values[i] = 0
    
    #Quitamos asignaturas que hayan sido convalidadas y tengan nota 0 
    dataHistCompleto['forma_aprobacion'].replace("HOMOLOGACIÓN POR VALIDACIÓN DE CONOCIMIENTOS","APROBADO",inplace = True)
    dataHistCompleto['forma_aprobacion'].replace("EQUIVALENCIA","CONVALIDACION",inplace = True)    
    dataHistCompleto['forma_aprobacion'].replace("HOMOLOGACION","CONVALIDACION",inplace = True)
    dataActual['forma_aprobacion'].replace("HOMOLOGACIÓN POR VALIDACIÓN DE CONOCIMIENTOS","CONVALIDACION",inplace = True)
    dataActual['forma_aprobacion'].replace("EQUIVALENCIA","CONVALIDACION",inplace = True)    
    dataActual['forma_aprobacion'].replace("HOMOLOGACION","CONVALIDACION",inplace = True)
    dataHastaActual['forma_aprobacion'].replace("HOMOLOGACIÓN POR VALIDACIÓN DE CONOCIMIENTOS","CONVALIDACION",inplace = True)
    dataHastaActual['forma_aprobacion'].replace("EQUIVALENCIA","CONVALIDACION",inplace = True)    
    dataHastaActual['forma_aprobacion'].replace("HOMOLOGACION","CONVALIDACION",inplace = True)
    
    #Se eliminan asignaturas convalidadas y nota final 0
    dataHistCompleto = dataHistCompleto.drop(dataHistCompleto[(dataHistCompleto['forma_aprobacion'] == 'CONVALIDACION') & (dataHistCompleto['nota_final'] == 0)].index)    
    dataHastaActual = dataHastaActual.drop(dataHastaActual[(dataHastaActual['forma_aprobacion'] == 'CONVALIDACION') & (dataHastaActual['nota_final'] == 0)].index)    
    dataActual = dataActual.drop(dataActual[(dataActual['forma_aprobacion'] == 'CONVALIDACION') & (dataActual['nota_final'] == 0)].index)
    #Se eliminan asignaturas reprobadas con nota final mayor de 70
    dataHistCompleto = dataHistCompleto.drop(dataHistCompleto[(dataHistCompleto['estado_asignatura'] == 'REPROBADO') & (dataHistCompleto['nota_final'] >= 70)].index)    
    dataHastaActual = dataHastaActual.drop(dataHastaActual[(dataHastaActual['estado_asignatura'] == 'REPROBADO') & (dataHastaActual['nota_final'] >= 70)].index)    
    dataActual = dataActual.drop(dataActual[(dataActual['estado_asignatura'] == 'REPROBADO') & (dataActual['nota_final'] >= 70)].index)
    #Se eliminan asignaturas en las que no hay registro de la actividad
    dataHistCompleto = dataHistCompleto.drop(dataHistCompleto[(dataHistCompleto['estado_asignatura'] == 'NO REGISTRADO') ].index)    
    dataHastaActual = dataHastaActual.drop(dataHastaActual[(dataHastaActual['estado_asignatura'] == 'NO REGISTRADO') ].index)    
    dataActual = dataActual.drop(dataActual[(dataActual['estado_asignatura'] == 'NO REGISTRADO') ].index)
    #Cogemos únicamente asignaturas con nota final mayor de 0
    dataHistCompleto = dataHistCompleto[dataHistCompleto['nota_final']>= 0]
    dataHastaActual = dataHastaActual[dataHastaActual['nota_final']>= 0]
    dataActual = dataActual[dataActual['nota_final']>= 0]
    #Asignaturas que el estudiante ha tomado	 
    dataTomadasHastaActual = dataHastaActual[dataHastaActual['forma_aprobacion'] != 'CONVALIDACION']    
    dataTomadasActual = dataActual[dataActual['forma_aprobacion'] != 'CONVALIDACION']
    #Semestre en el que estamos
    #semMax = dataTomadasActual.groupby(["id","codigo_carrera"],as_index = False).semestre_cursa.max()
    #dataTomadasActual = dataTomadasActual.merge(semMax, how = "inner", on = ["id","codigo_carrera","semestre_cursa"])
        
    #Numero de materias último  semestre
    numMatSem = dataTomadasActual.groupby(['id', 'codigo_carrera'],as_index = False).codigo_asignatura.count()
    
    #Quitamos materias que se están cursando actualmente para hacer demás calculos
    dataTomadasActual = dataTomadasActual[dataTomadasActual['estado_asignatura']!="CURSANDO"]
    dataActual = dataActual[dataActual['estado_asignatura']!="CURSANDO"]
    dataHastaActual = dataHastaActual[dataHastaActual['estado_asignatura']!="CURSANDO"]
    dataTomadasHastaActual = dataTomadasHastaActual[dataTomadasHastaActual['estado_asignatura']!="CURSANDO"]
    dataHistCompleto = dataHistCompleto[dataHistCompleto['estado_asignatura']!="CURSANDO"]
    
    #Calculamos numero de materias aprobadas por estudiante en todos sus semestres    
    aprobadasTot =  dataHistCompleto[(dataHistCompleto['estado_asignatura'] == 'APROBADO') & (dataHistCompleto['nota_final'] != 0)]
    numMateriasApTot = aprobadasTot.groupby(['id', 'codigo_carrera'],as_index = False).codigo_asignatura.count()
    numMateriasApTot.columns = ["id","codigo_carrera","Count"]
    #Número de segundas matrículas aprobadas en total hasta este semestre
    aprobSegData = dataHastaActual[(dataHastaActual['numero_matricula'] == 2) & (dataHastaActual['estado_asignatura'] == 'APROBADO')]
    aprobSegDataCount = aprobSegData.groupby(['id', 'codigo_carrera'], as_index=False).codigo_asignatura.count()
    #Número de segundas matrículas hasta este semestre 
    aprobSegTot = dataHastaActual[(dataHastaActual['numero_matricula'] == 1) & (dataHastaActual['estado_asignatura'] == 'REPROBADO')]
    aprobSegTotCount = aprobSegTot.groupby(['id', 'codigo_carrera'], as_index=False).codigo_asignatura.count()
    #Numero de veces que un alumno ha aprobado en 3º convocatoria (si reprueban los echan) en total hasta ahora
    reprobSegTotData = dataHastaActual[(dataHastaActual['numero_matricula'] == 3) & (dataHastaActual['estado_asignatura'] == 'APROBADO')]
    reprobSegTotDataCount = reprobSegTotData.groupby(['id', 'codigo_carrera'], as_index=False).codigo_asignatura.count()
    #Numero de veces que un alumno ha llegado a ultima matricula hasta ahora
    reprobSegTot = dataHastaActual[(dataHastaActual['numero_matricula'] == 2) & (dataHastaActual['estado_asignatura'] == 'REPROBADO')]
    reprobSegTotCount = reprobSegTot.groupby(['id', 'codigo_carrera'],as_index = False).codigo_asignatura.count()
    
    #Segundas matriculas actuales
    reprobSegActual = aprobSegTotCount.merge(aprobSegDataCount, on=['id','codigo_carrera'], how = 'outer')
    reprobSegActual = reprobSegActual.merge(reprobSegTotDataCount, on=['id','codigo_carrera'], how = 'outer')
    reprobSegActual = reprobSegActual.merge(reprobSegTotCount, on=['id','codigo_carrera'], how = 'outer').fillna(0)
    reprobSegActual.columns = ['id','codigo_carrera','segMatActual','uno','dos','tres']
    reprobSegActual['segMatActual'] = reprobSegActual['segMatActual'] - reprobSegActual['uno'] - reprobSegActual['dos'] - reprobSegActual['tres']
    reprobSegActual = reprobSegActual[['id','codigo_carrera','segMatActual']]
    reprobSegActual[reprobSegActual['segMatActual'] < 0] = 0
    
    #Numero de veces que un alumno ha llegado a 3º convocatoria (si reprueban los echan) este semestre
    alum = cambioEstudiante[['id','codigo_carrera']]
    reprobSeg = dataHastaActual[(dataHastaActual['numero_matricula'] == 2) & (dataHastaActual['estado_asignatura'] == 'REPROBADO')]
    reprobSegCount = reprobSeg.groupby(['id', 'codigo_carrera'], as_index=False).codigo_asignatura.count()
    reprobSegCount = reprobSegCount.merge(alum, on=['id','codigo_carrera'], how = 'outer').fillna(0)
    reprobSegCount = reprobSegCount.drop_duplicates(subset="id", keep = "first")

    terMatAct = reprobSegCount.merge(reprobSegTotDataCount, on=['id','codigo_carrera'], how = 'outer').fillna(0)
    terMatAct = terMatAct[['id','codigo_carrera','codigo_asignatura_y','codigo_asignatura_x']]
    terMatAct.columns = ['id','codigo_carrera','terMatActual','uno']
    terMatAct['terMatActual'] = terMatAct['terMatActual'] - terMatAct['uno']
    terMatAct = terMatAct[['id','codigo_carrera','terMatActual']]
    terMatAct[terMatAct['terMatActual'] < 0] = 0
    terMatAct = terMatAct.drop_duplicates(subset="id", keep = "first")
    
    dataHastaActual['nota_final'] = dataHastaActual['nota_final'].astype(float)
    
    #Calculamos numero de suspensos por estudiante hasta el momento
    fail =  dataTomadasHastaActual[dataTomadasHastaActual['estado_asignatura'] == 'REPROBADO']
    fail =  fail[fail['nota_final'] < 70]
    numFail = fail.groupby(['id','codigo_carrera'],as_index = False).codigo_asignatura.count()
    numFail.columns = ['id','codigo_carrera',"Count"]
    
    #Suma de todas las notas finales REPROBADAS hasta el momento
    sumRp = fail.groupby(['id', 'codigo_carrera'],as_index = False).nota_final.sum()
    sumRp['notafinal'] = sumRp['nota_final'].fillna(0)
    
    #Media de notas unicamente utilizando las notas REPROBADAS hasta el momento
    dataMediaRp = pd.merge(sumRp, numFail, how='outer', on=['id','codigo_carrera'])  
    dataMediaRp['notafinal'] = dataMediaRp['notafinal'].fillna(0)
    dataMediaRp['media'] = dataMediaRp.notafinal / dataMediaRp.Count
    dataMediaRp['media'] = dataMediaRp['media'].fillna(0)
    dataMediaRp = pd.DataFrame(dataMediaRp).drop("nota_final",1).drop("notafinal",1).drop("Count",1)
    
        
    #Calculamos numero de anuladas por estudiante hasta el momento
    anuladas =  dataHastaActual[dataHastaActual['estado_asignatura'] == 'ANULADO']
    numAnuladas = anuladas.groupby(['id', 'codigo_carrera'],as_index = False).codigo_asignatura.count()	
    numAnuladas.columns = ['id','codigo_carrera',"Count"]
    
    #Calculamos numero de materias tomadas total por estudiante hasta el momento
    numMateriasEst = dataTomadasHastaActual.groupby(['id', 'codigo_carrera'],as_index = False).codigo_asignatura.count()	
    numMateriasEst.columns = ['id','codigo_carrera',"Count"]
    
    #Materias que está cursado el estudiante en este momento
    cursando =  dataActual[dataActual['estado_asignatura'] == 'CURSANDO']
    
    #Calculamos numero de materias cursando por estudiante ese semestre
    numMateriasCurs = cursando.groupby(['id', 'codigo_carrera'],as_index = False).codigo_asignatura.count()
    numMateriasCurs.columns = ["id","codigo_carrera","Count"]
    
    #Materias aprobadas y con nota final hasta el momento
    aprobadas =  dataTomadasHastaActual[(dataTomadasHastaActual['estado_asignatura'] == 'APROBADO') & (dataTomadasHastaActual['nota_final'] != 0)]    
    
    #Calculamos numero de materias aprobadas por estudiante ese semestre
    numMateriasAp = aprobadas.groupby(['id', 'codigo_carrera'],as_index = False).codigo_asignatura.count()
    numMateriasAp.columns = ["id","codigo_carrera","Count"]
    
    #Suma de todas las notas finales APROBADAS hasta el momento
    sumAp = aprobadas.groupby(['id', 'codigo_carrera'],as_index = False).nota_final.sum()
    sumAp['notafinal'] = sumAp['nota_final'].fillna(0)
    
    #Media de notas unicamente utilizando las notas aprobadas hasta el momento
    dataMediaAp = pd.merge(sumAp, numMateriasAp, how='outer', on=['id','codigo_carrera'])  
    dataMediaAp['notafinal'] = dataMediaAp['notafinal'].fillna(0)
    dataMediaAp['media'] = dataMediaAp.notafinal / dataMediaAp.Count
    dataMediaAp['media'] = dataMediaAp['media'].fillna(0)
    dataMediaAp = dataMediaAp.drop("nota_final",1).drop("Count",1).drop("notafinal",1)        
    
    #Suma de todas las notas finales REPROBADAS hasta el momento
    sumRp = fail.groupby(['id', 'codigo_carrera'],as_index = False).nota_final.sum()
    
    #Notas finales de asignaturas aprobadas y reprobadas total
    sumApRp = pd.merge(sumRp, sumAp, how='outer', on=['id','codigo_carrera']) 
    sumApRp = sumApRp.fillna(0)
    sumApRp['notafinal'] = sumApRp['nota_final_x'] + sumApRp['nota_final_y']
    sumApRp = sumApRp.fillna(0).drop('nota_final_x',1).drop('nota_final_y',1)
    sumApRp = pd.DataFrame(sumApRp)
    sumApRp.columns = ["id","codigo_carrera","notafinal"]    
    
    #Asignaturas aprobadas y reprobadas hasta el semestre actual
    aprobadasReprobadas =  dataHastaActual[dataHastaActual['estado_asignatura'] != 'ANULADO']
    aprobadasReprobadas = aprobadasReprobadas[((aprobadasReprobadas['estado_asignatura'] == 'APROBADO')& (aprobadasReprobadas['nota_final'] != 0)) | (aprobadasReprobadas['estado_asignatura'] == 'REPROBADO') ]
    
    #Calculamos numero de materias aprobadas y reprobadas por estudiante hasta el semestre actual
    numMateriasApRp = aprobadasReprobadas.groupby(['id', 'codigo_carrera'], as_index = False).codigo_asignatura.count()
    numMateriasApRp.columns = ["id","codigo_carrera","Count"]
    
    #Media de notas utilizando las notas aprobadas y reprobadas hasta el semestre actual
    dataMediaApRp = pd.merge(sumApRp, numMateriasApRp, how='outer', on=['id','codigo_carrera'])
    dataMediaApRp = dataMediaApRp.fillna(0)  
    dataMediaApRp['media'] = dataMediaApRp.notafinal / dataMediaApRp.Count
    dataMediaApRp['media'] = dataMediaApRp['media'].fillna(0)
    dataMediaApRp = dataMediaApRp.drop("notafinal",1).drop("Count",1)
    
    #Materias aprobadas y con nota final en el momento
    aprobadasActual =  dataTomadasActual[(dataTomadasActual['estado_asignatura'] == 'APROBADO') & (dataTomadasActual['nota_final'] != 0)]    
    
    #Suma de todas las notas finales APROBADAS hasta el momento
    sumApActual = aprobadasActual.groupby(['id', 'codigo_carrera'],as_index = False).nota_final.sum()
    sumApActual['notafinal'] = sumApActual['nota_final'].fillna(0)    
    
    #Calculamos numero de suspensos por estudiante hasta el momento
    failActual =  dataTomadasActual[dataTomadasActual['estado_asignatura'] == 'REPROBADO']
    failActual =  failActual[failActual['nota_final'] < 70]    
    
    #Suma de todas las notas finales REPROBADAS hasta el momento
    sumRpActual = failActual.groupby(['id', 'codigo_carrera'],as_index = False).nota_final.sum()
    
    #Notas finales de asignaturas aprobadas y reprobadas total
    sumApRpActual = pd.merge(sumRpActual, sumApActual, how='outer', on=['id','codigo_carrera']) 
    sumApRpActual = sumApRpActual.fillna(0)
    sumApRpActual['notafinal'] = sumApRpActual['nota_final_x'] + sumApRpActual['nota_final_y']
    sumApRpActual = sumApRpActual.fillna(0).drop('nota_final_x',1).drop('nota_final_y',1)
    sumApRpActual = pd.DataFrame(sumApRpActual)
    sumApRpActual.columns = ["id","codigo_carrera","notafinal"]
    
    #Asignaturas aprobadas y reprobadas en el semestre actual 
    aprobadasReprobadasActual =  dataActual[dataActual['estado_asignatura'] != 'ANULADO']
    aprobadasReprobadasActual = aprobadasReprobadasActual[((aprobadasReprobadasActual['estado_asignatura'] == 'APROBADO')& (aprobadasReprobadasActual['nota_final'] != 0)) | (aprobadasReprobadasActual['estado_asignatura'] == 'REPROBADO') ]
    
    #Calculamos numero de materias aprobadas y reprobadas en el semestre actual
    numMateriasApRpActual = aprobadasReprobadasActual.groupby(['id', 'codigo_carrera'], as_index = False).codigo_asignatura.count()
    numMateriasApRpActual.columns = ["id","codigo_carrera","Count"]
    
    #Media de notas utilizando las notas aprobadas y reprobadas en el semestre
    dataMediaApRpActual = pd.merge(sumApRpActual, numMateriasApRpActual, how='outer', on=['id','codigo_carrera'])
    dataMediaApRpActual = dataMediaApRpActual.fillna(0)  
    dataMediaApRpActual['media'] = dataMediaApRpActual.notafinal / dataMediaApRpActual.Count
    dataMediaApRpActual['media'] = dataMediaApRpActual['media'].fillna(0)
    dataMediaApRpActual = dataMediaApRpActual.drop("notafinal",1).drop("Count",1)  
    
    #Calculamos numero de materias aprobadas por estudiante ese semestre
    numMateriasApActual = aprobadasActual.groupby(['id', 'codigo_carrera'],as_index = False).codigo_asignatura.count()
    numMateriasApActual.columns = ["id","codigo_carrera","Count"]   
    
    #Calculamos numero de materias tomadas total por estudiante en el momento
    numMateriasEstActual = dataTomadasActual.groupby(['id', 'codigo_carrera'],as_index = False).codigo_asignatura.count()	
    numMateriasEstActual.columns = ['id','codigo_carrera',"Count"]    
    
    #Rate aprobadas semestre actual
    aprobCursadosActual = pd.merge(numMateriasEstActual,numMateriasApActual,how = "outer", on=["id","codigo_carrera"])
    aprobCursadosActual = aprobCursadosActual.fillna(0)
    aprobCursadosActual['Count'] = aprobCursadosActual['Count_y']/aprobCursadosActual['Count_x']
    aprobCursadosActual = aprobCursadosActual.drop("Count_x",1).drop("Count_y",1)

    
    #Años que lleva un estudiante en la universidad desde que empezó hasta que cursó por última vez
    #Sin tener en cuenta si lo ha dejado un tiempo de por medio
    datamax = dataHastaActual.groupby(['id','codigo_carrera'],as_index = False).anio_curso_asignatura.max()
    datamin = dataHastaActual.groupby(['id','codigo_carrera'],as_index = False).anio_curso_asignatura.min()
    añosUni = pd.merge(datamax, datamin, how='outer', on=['id','codigo_carrera'])
    añosUni.anio_curso_asignatura_x = (añosUni.anio_curso_asignatura_x).astype(int)
    añosUni.anio_curso_asignatura_y = (añosUni.anio_curso_asignatura_y).astype(int)
    añosUni.anio_curso_asignatura_y = añosUni.anio_curso_asignatura_x - añosUni.anio_curso_asignatura_y + 1
    añosUni = añosUni.drop("anio_curso_asignatura_y",1)
    
    #Media ponderada de cada asignatura de cada estudiante
    dataHastaActual = dataHastaActual[dataHastaActual['total_horas_asignatura'] != 0]
    dataHastaActual['numero_matricula'].replace(2,0.9,inplace = True)
    dataHastaActual['numero_matricula'].replace(3,0.7,inplace = True)
    dataHastaActual['numero_matricula'].replace(4,0.7,inplace = True)
    dataHastaActual['numero_matricula'].replace(5,0.7,inplace = True)
    dataHastaActual['numero_matricula'].replace(6,0.7,inplace = True)
    
    #25horas = 1 credito
    dataHastaActual['numero_matricula'] = (dataHastaActual['numero_matricula']*(dataHastaActual.nota_final).astype(float)*dataHastaActual['total_horas_asignatura']).astype(int)
    
    #Empezamos a calcular la media ponderada (formula UC3M) hasta ahora
    dataS =  dataHastaActual[dataHastaActual['estado_asignatura'] == 'APROBADO']
    dataS =  dataS[dataS.nota_final != 0]
   
    #Todos las horas cursados por estudiante (AN, RP Y AP) hasta ahora
    credCursados = dataHastaActual.groupby(['id','codigo_carrera'],as_index = False).total_horas_asignatura.sum()#horas cursados
    
    #Numero de horas aprobados por carrera y estudiante
    dataP =  dataHistCompleto[dataHistCompleto['estado_asignatura'] == 'APROBADO']
    credAprob = dataP.groupby(['id','codigo_carrera'],as_index = False).total_horas_asignatura.sum()#horas aprobados
     	
    #aprobCursados = credAprob / credCursados
    aprobCursados = pd.merge(numMateriasEst,numMateriasAp,how = "outer", on=["id","codigo_carrera"])
    aprobCursados = aprobCursados.fillna(0)
    aprobCursados['Count'] = aprobCursados['Count_y']/aprobCursados['Count_x']
    aprobCursados = aprobCursados.drop("Count_x",1).drop("Count_y",1)

    #Ratio de asignaturas que ha anulado frente a las matriculadas en total
    rateAnuladaCursadas = pd.merge(numMateriasEst,numAnuladas,how = "outer", on=["id","codigo_carrera"])
    rateAnuladaCursadas = rateAnuladaCursadas.fillna(0)
    rateAnuladaCursadas ['Count'] = rateAnuladaCursadas['Count_y'] / rateAnuladaCursadas['Count_x']
    rateAnuladaCursadas = rateAnuladaCursadas.drop("Count_x",1).drop("Count_y",1)
    
    #Ratio de asignaturas que esta cursando frente a las matriculadas en total
    rateCursando = pd.merge(numMateriasEst,numMateriasCurs,how = "outer", on=["id","codigo_carrera"])
    rateCursando = rateCursando.fillna(0)
    rateCursando ['Count'] = rateCursando['Count_y'] / rateCursando['Count_x']
    rateCursando = rateCursando.drop("Count_x",1).drop("Count_y",1)
    
    
    #Ratio de asignaturas que ha reprobado frente a las matriculadas en total
    rateReprobadasCursadas = pd.merge(numMateriasEst,numFail,how = "outer", on=["id","codigo_carrera"])
    rateReprobadasCursadas = rateReprobadasCursadas.fillna(0)
    rateReprobadasCursadas ['Count'] = rateReprobadasCursadas['Count_y'] / rateReprobadasCursadas['Count_x']
    rateReprobadasCursadas = rateReprobadasCursadas.drop("Count_x",1).drop("Count_y",1)
    
    #Nota media total a falta de dividir por el numero de horas
    notPond = dataS.groupby(['id','codigo_carrera'],as_index = False).numero_matricula.sum()#Suma de las notas poderadas
    #notPond = notPond[notPond.call != 0]
    
    #Dividimos media ponderada y creditos aprobados
    dataMedia = pd.merge(notPond, credAprob,  how='outer', on=['id','codigo_carrera'])
    dataMedia.total_horas_asignatura = dataMedia.numero_matricula / dataMedia.total_horas_asignatura
    dataMedia = dataMedia.drop("numero_matricula",1)
    
    #Numero de años sin matricularse
    ultMat = dataHistCompleto.groupby(['id','codigo_carrera'],as_index = False).anio_curso_asignatura.max()
    añoActual = int(time.strftime("%Y"))
    ultMat['anio_curso_asignatura'] = añoActual  - ultMat.anio_curso_asignatura.values
    
    dataMedia = pd.merge(dataMedia, ultMat,how = 'outer', on = ['id','codigo_carrera'])
    dataMedia = dataMedia.fillna(0)
    dataMedia = pd.merge(dataMedia,rateReprobadasCursadas ,how = 'outer', on = ['id','codigo_carrera'])
    dataMedia = pd.merge(dataMedia,rateAnuladaCursadas ,how = 'outer', on = ['id','codigo_carrera'])
    dataMedia = pd.merge(dataMedia,rateCursando ,how = 'outer', on = ['id','codigo_carrera'])
    dataMedia = pd.merge(dataMedia,aprobCursados ,how = 'outer', on = ['id','codigo_carrera'])
    dataMedia = pd.merge(dataMedia,dataMediaApRp,how = 'outer', on = ['id','codigo_carrera'])
    dataMedia = pd.merge(dataMedia,dataMediaAp ,how = 'outer', on = ['id','codigo_carrera'])
    dataMedia = pd.merge(dataMedia,terMatAct ,how = 'outer', on = ['id','codigo_carrera'])
    dataMedia = pd.merge(dataMedia,reprobSegCount ,how = 'outer', on = ['id','codigo_carrera'])
    dataMedia = pd.merge(dataMedia,numMatSem ,how = 'outer', on = ['id','codigo_carrera'])
    dataMedia = pd.merge(dataMedia,aprobSegTotCount ,how = 'outer', on = ['id','codigo_carrera'])
    dataMedia = pd.merge(dataMedia,reprobSegActual ,how = 'outer', on = ['id','codigo_carrera'])
    dataMedia = pd.merge(dataMedia,dataMediaRp ,how = 'outer', on = ['id','codigo_carrera'])    
    dataMedia = pd.merge(dataMedia,dataMediaApRpActual ,how = 'outer', on = ['id','codigo_carrera'])
    dataMedia = pd.merge(dataMedia,aprobCursadosActual ,how = 'outer', on = ['id','codigo_carrera'])
    dataMedia = dataMedia.fillna(0)
    
    dataMedia.columns = ["id","codigo_carrera","mediaPond","ultMatricula","rateReprobadas","rateAnuladas","rateCursando","rateAprobadas","mediaAPRP","mediaAP","terMatActual","terMatTot","numMatSem","segMatTot","segMatActual","mediaRP","mediaAPRPActual","rateAprobadasActual"]
       
    #Numero de creditos aprobados
    dataCredPass = pd.merge(credAprob, añosUni, how='outer', on=['id','codigo_carrera'])
    dataCredPass = pd.merge(dataCredPass, numMateriasApTot, how='outer', on=['id','codigo_carrera'])
    dataCredPass = dataCredPass.fillna(0)
    dataCredPass.columns = ["id","codigo_carrera","numero_horas_aprobados","anio_curso_asignatura","Materias aprobadas"]
        
    return dropout,dataMedia,dataCredPass