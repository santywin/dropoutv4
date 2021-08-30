import psycopg2, psycopg2.extras
import pandas as pd
import numpy as np

def download():    
    
    conn = psycopg2.connect(database='lala',user='lala',password='PB2Cx3fDEgfFTpPn', host='172.16.101.55')

    cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)

    print("Bajando datos de estudiantes")
    cur.execute("SELECT * FROM estudiantes")# ORDER BY id")
    data =cur.fetchall();
    df = pd.DataFrame(np.array(data), columns = data[0].keys())
    #df.drop("codigo_malla",1,inplace = True)
    #df.to_csv('dim_estudiante.csv')

    print("Bajando calificaciones de estudiantes")
    cur.execute("SELECT * FROM calificaciones")# ORDER BY id")
    data =cur.fetchall();
    df1 = pd.DataFrame(np.array(data), columns = data[0].keys())    
    df1 = df1.sort_values(["id"])
    #df1.to_csv('dim_historial_estudiante.csv')
    """
    ####Ponemos los ids#############
    df['id'] = range(len(df))
    """
    """
    df1['id'] = 0
    id = 1
    for i in range(len(df1)):
        if df1['codigo_carrera'].values[i] is not None :
            df1['id'].values[i] = id
        elif df1['codigo_carrera'].values[i-1] is not None:
            id = id + 1
            #calificaciones.drop(calificaciones.index[i])
    """
    df1 = df1.dropna(subset=['codigo_carrera'])
    
    return df,df1
