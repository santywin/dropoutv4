# -*- coding: utf-8 -*-
"""
Created on Fri Mar 13 12:38:07 2020

@author: Jon Imaz
"""

#Número de segundas matrículas aprobadas en total hasta este semestre
aprobSegData = dataT[(dataT['numero_matricula'] == 2) & (dataT['estado_asignatura'] == 'APROBADO')]
aprobSegDataCount = aprobSegData.groupby(['id', 'codigo_carrera'], as_index=False).codigo_asignatura.count()
#Número de segundas matrículas hasta este semestre 
aprobSegTot = dataT[(dataT['numero_matricula'] == 1) & (dataT['estado_asignatura'] == 'REPROBADO')]
aprobSegTotCount = aprobSegTot.groupby(['id', 'codigo_carrera'], as_index=False).codigo_asignatura.count()
#Numero de veces que un alumno ha aprobado en 3º convocatoria (si reprueban los echan) en total hasta ahora
reprobSegTotData = dataT[(dataT['numero_matricula'] == 3) & (dataT['estado_asignatura'] == 'APROBADO')]
reprobSegTotDataCount = reprobSegTotData.groupby(['id', 'codigo_carrera'], as_index=False).codigo_asignatura.count()
#Numero de veces que un alumno ha llegado a ultima matricula hasta ahora
reprobSegTot = dataT[(dataT['numero_matricula'] == 2) & (dataT['estado_asignatura'] == 'REPROBADO')]
reprobSegTotCount = reprobSegTot.groupby(['id', 'codigo_carrera'],as_index = False).codigo_asignatura.count()

#Segundas matriculas actuales
reprobSegActual = aprobSegTotCount.merge(aprobSegDataCount, on=['id','codigo_carrera'], how = 'outer')
reprobSegActual = reprobSegActual.merge(reprobSegTotDataCount, on=['id','codigo_carrera'], how = 'outer')
reprobSegActual = reprobSegActual.merge(reprobSegTotCount, on=['id','codigo_carrera'], how = 'outer').fillna(0)
reprobSegActual.columns = ['id','codigo_carrera','segMatActual','uno','dos','tres']
reprobSegActual['segMatActual'] = reprobSegActual['segMatActual'] - reprobSegActual['uno'] - reprobSegActual['dos'] - reprobSegActual['tres']
reprobSegActual = reprobSegActual[['id','codigo_carrera','segMatActual']]


#Numero de veces que un alumno ha llegado a 3º convocatoria (si reprueban los echan) este semestre
alum = cambioEstudiante[['id','codigo_carrera']]
reprobSeg = dataT[(dataT['numero_matricula'] == 2) & (dataT['estado_asignatura'] == 'REPROBADO')]
reprobSegCount = reprobSeg.groupby(['id', 'codigo_carrera'], as_index=False).codigo_asignatura.count()
reprobSegCount = reprobSegCount.merge(alum, on=['id','codigo_carrera'], how = 'outer').fillna(0)
terMatAct = reprobSegCount.merge(reprobSegTotDataCount, on=['id','codigo_carrera'], how = 'outer').fillna(0)
terMatAct.columns = ['id','codigo_carrera','terMatActual','uno']
terMatAct['terMatActual'] = terMatAct['terMatActual'] - terMatAct['uno']
terMatAct = terMatAct[['id','codigo_carrera','terMatActual']]


