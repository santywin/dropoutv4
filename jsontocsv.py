import csv
import json
import pandas as pd 

def formateo():
    #Cambiamos de formato las calificaciones de los estudiantes
    est = json.loads(open('estudiantes_sin_malla.json').read())
    
    f = csv.writer(open("estudiantes.csv", "w+"))
    
    
    # Write CSV Header, If you dont need that, remove this line
    """
    for y in range(len(est)):	
    	est[y]['id'] = y+1
    """
    f.writerow(["id","codigo_carrera",  "genero","nombre_carrera","total_horas_carrera"])
    
    for x in est:
        f.writerow([x["id"],
                    x["codigo_carrera"],
                    #x["codigo_malla"],
                    x["genero"],
                    x["nombre_carrera"],
                    x["total_horas_carrera"]])
    file = 'estudiantes.csv'
    df = pd.read_csv(file,encoding =  "ISO-8859-1")
    #df.to_csv("estudiantes1.csv")
    
    #Cambiamos de formato las calificaciones de los estudiantes
    calif = json.loads(open('calificaciones.json').read())
    
    f = csv.writer(open("calificaciones.csv", "w+"))
    """
    for y in range(len(calif)):
    	for x in range(len(calif[y])):
    		calif[y][x]['id'] = y+1
    """
    # Write CSV Header, If you dont need that, remove this line
    
    f.writerow(['id','anio_curso_asignatura', 'codigo_asignatura', 'codigo_carrera', 'codigo_curriculum', 'estado_asignatura', 'forma_aprobacion',  'nota_final', 'numero_creditos', 'numero_matricula', 'total_horas_asignatura'])
    
    for y in calif:
    	f.writerow("")
    	for x in y:
    		f.writerow([x["codigo_estudiante"],
                x["anio_curso_asignatura"],
        		x["codigo_asignatura"],
        		x["codigo_carrera"],
                x["codigo_curriculum"],
                x["estado_asignatura"],
                x["forma_aprobacion"],
                x["nota_final"],
                x["numero_creditos"],
                x["numero_matricula"],
                x["total_horas_asignatura"]])
    file = 'calificaciones.csv'
    df1 = pd.read_csv(file,encoding =  "ISO-8859-1")
    #df1.to_csv("calificaciones1.csv")
    
    return df, df1
    """
    grad = json.loads(open('graduados.json').read())
    
    df = pd.DataFrame(grad)
    df.columns = ['dict']
    
    arr = np.array(df['dict']).tolist()
    arr = [{'CARRERA_ID': 0, 'NOMBRE': 0, 'DURACION_ANIOS': 0, 'ANIO_INGRESO': 0, 'SEMESTRE_INGRESO': 0, 'SEMESTRE_INGRESO_DESC': 0, 'ANIO_EGRESO': 0, 'SEMESTRE_EGRESO': 115, 'SEMESTRE_EGRESO_DESC': 'SEPTIEMBRE 2016-FEBRERO 2017', 'NOTA_FINAL': 0} if v is None else v for v in arr]
    
    df = pd.DataFrame.from_records(arr)
    
    df['id'] = 1
    arr = np.array(df)
    for i in range(len(arr)):
    	arr[i][10] = i+1
    df = pd.DataFrame(arr, columns=df.columns)
    df = df[df.CARRERA_ID !=0]
    df.to_csv("graduados.csv")
    
    
    
    mallas = json.loads(open('mallas.json').read())
    
    f = csv.writer(open("mallas.csv", "w+"))
    # Write CSV Header, If you dont need that, remove this line
    
    f.writerow(['MALLA_ID', 'CARRERA', 'MALLA_ANIO', 'SEMESTRE', 'ASIGNATURA_CODIGO', 'NOMBRE_ASIGNATURA', 'CREDITOS', 'TOTAL_HORAS_CICLO', 'EJE_FORMACION', 'OPTATIVO', 'ELECTIVO','TOTAL_HORAS_MALLA'])
    
    for x in mallas:
        f.writerow([x["MALLA_ID_"],
                    x["CARRERA"],
                    x["MALLA_ANIO"],
                    x["SEMESTRE"],
                    x["ASIGNATURA_CODIGO"],
                    x["NOMBRE_ASIGNATURA"],
                    x["CREDITOS"],
                    x["TOTAL_HORAS_CICLO"],
                    x["EJE_FORMACION"],
                    x["OPTATIVO"],
                    x["ELECTIVO"],
                    x["TOTAL_HORAS_MALLA"]])
    
    file = 'mallas.csv'
    df = pd.read_csv(file,encoding =  "ISO-8859-1")
    df = df.drop_duplicates()
    df['id'] = 1
    arr = np.array(df)
    for i in range(len(arr)):
    	arr[i][12] = i+1
    df = pd.DataFrame(arr, columns=df.columns)
    df.to_csv("mallas1.csv")
    
    
    
    pga = json.loads(open('pga.json').read())
    
    f = csv.writer(open("pga.csv", "w+"))
    
    for y in range(len(pga)):
    	for x in range(len(pga[y])):
    		pga[y][x]['id'] = y+1
    	if len(pga[y]) == 0:
    		pga[y] =  [{"CARRERA":0, "PERLEC_ID":0, "DESCRIPCIONPERIODO":0,"PGA":0,"id":y+1}]
    
    # Write CSV Header, If you dont need that, remove this line
    f.writerow(["CARRERA", "PERLEC_ID", "DESCRIPCIONPERIODO","PGA","id"])
    
    for y in pga:
    	for x in y:
    		f.writerow([x["CARRERA"],
                    x["PERLEC_ID"],
                    x["DESCRIPCIONPERIODO"],
                    x["PGA"],
                    x["id"]])
    
    
    file = 'pga.csv'
    df = pd.read_csv(file)
    df.to_csv("pga1.csv")
    """