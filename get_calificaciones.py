import pandas as pd

def cambCal(calificaciones):

    df = calificaciones.sort_values(by='id', ascending=True)
    df.drop("semestre_en_carrera",1,inplace = True)
    
    #Uno de los dos campos ID no hace falta
    #df.drop("ID",1,inplace = True)
    #df.drop("estudiante_id",1,inplace = True)  
    
    #Uno de los dos campos ID no hace falta
    #df.drop("CODIGO_CARRERA",1,inplace = True)
    df.drop("codigo_curriculum",1,inplace = True) 
    
    df['estado_asignatura'].replace("RETIRADO","REPROBADO",inplace = True)
    
    return df