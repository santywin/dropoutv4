import pandas as pd


#def main():

def cambioEst(estudiantes):
       
    df = estudiantes.sort_values(by='id', ascending=True)
    df['graduado'].replace("S","1", inplace = True)
    df['graduado'].replace("N","2", inplace = True)
   
    return df
    