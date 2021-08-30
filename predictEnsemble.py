import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV
import Ensembling as ensemble
import Useful_plots as uplot
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.calibration import calibration_curve


def predict(final,carrera, i):
    
    #file = 'Final.csv'
    #data = pd.read_csv(file)
    data = final.copy()
    dataPred = data[data.nombre_carrera == carrera]
        
    #dataPred = data.drop("id",1).drop("degree",1)
    #dataPred = data.drop("Unnamed: 0",1)
    dataPred['dropout'].fillna("",inplace = True)
    dataPred['dropout'] = dataPred['dropout'].astype(str)
    dataPred['dropout'] = dataPred['dropout'].str.strip()    
    dataPred['dropout'].fillna("2",inplace = True)
    dataPred['dropout'].replace("","2",inplace = True)
    dataPred = dataPred.fillna(0)
    dataTrain = dataPred[dataPred.dropout != "2"]
    #Eliminamos estudiantes que meten ruido
    dataTrain.dropout = dataTrain.dropout.astype(str)
    mediaCarrera = dataTrain.groupby(['nombre_carrera']).mediaAP.sum().to_frame()/dataTrain.groupby(['nombre_carrera']).mediaAP.count().to_frame()
    mediaCarreraRateP = dataTrain.groupby(['nombre_carrera']).rateAprobadas.sum().to_frame()/dataTrain.groupby(['nombre_carrera']).rateAprobadas.count().to_frame()
    
    aprobados = dataTrain[dataTrain['dropout'] == "1"]
    abandonados = dataTrain[dataTrain['dropout'] == "0"]
    abandonados = abandonados[abandonados['mediaAP'] != 0]
    mediaCarreraAPGrad = aprobados.groupby(['nombre_carrera']).mediaAP.sum().to_frame()/aprobados.groupby(['nombre_carrera']).mediaAP.count().to_frame()
    mediaCarreraAPAbandon = abandonados.groupby(['nombre_carrera']).mediaAP.sum().to_frame()/abandonados.groupby(['nombre_carrera']).mediaAP.count().to_frame()
    mediaCarreraAPRPGrad = aprobados.groupby(['nombre_carrera']).mediaAPRP.sum().to_frame()/aprobados.groupby(['nombre_carrera']).mediaAPRP.count().to_frame()
    mediaCarreraAPRPAbandon = abandonados.groupby(['nombre_carrera']).mediaAPRP.sum().to_frame()/abandonados.groupby(['nombre_carrera']).mediaAPRP.count().to_frame()
    mediaCarreraRPGrad = aprobados.groupby(['nombre_carrera']).mediaRP.sum().to_frame()/aprobados.groupby(['nombre_carrera']).mediaRP.count().to_frame()
    mediaCarreraRPAbandon = abandonados.groupby(['nombre_carrera']).mediaRP.sum().to_frame()/abandonados.groupby(['nombre_carrera']).mediaRP.count().to_frame()
    mediaCarreraSecRPGrad = aprobados.groupby(['nombre_carrera']).terMatTot.sum().to_frame()/aprobados.groupby(['nombre_carrera']).terMatTot.count().to_frame()
    mediaCarreraSecRPAbandon = abandonados.groupby(['nombre_carrera']).terMatTot.sum().to_frame()/abandonados.groupby(['nombre_carrera']).terMatTot.count().to_frame()
    
    #Media rates
    mediaCarreraRateRPGrad = aprobados.groupby(['nombre_carrera']).rateReprobadas.sum().to_frame()/aprobados.groupby(['nombre_carrera']).rateReprobadas.count().to_frame()
    mediaCarreraRateRPAbandon = abandonados.groupby(['nombre_carrera']).rateReprobadas.sum().to_frame()/abandonados.groupby(['nombre_carrera']).rateReprobadas.count().to_frame()
    mediaCarreraRateAPGrad = aprobados.groupby(['nombre_carrera']).rateAprobadas.sum().to_frame()/aprobados.groupby(['nombre_carrera']).rateAprobadas.count().to_frame()
    mediaCarreraRateAPAbandon = abandonados.groupby(['nombre_carrera']).rateAprobadas.sum().to_frame()/abandonados.groupby(['nombre_carrera']).rateAprobadas.count().to_frame()
    #dataTrain['mediaCarreraAP'] = mediaCarrera['mediaAP'].values[0]
        
    for i in range(len(dataTrain)):
        if  dataTrain['rateAprobadas'].values[i] >= mediaCarreraRateP['rateAprobadas'].values[0]*1.05 and dataTrain['dropout'].values[i] == "0" and dataTrain['mediaAP'].values[i] >= mediaCarrera['mediaAP'].values[0]*1.05:
            dataTrain['dropout'].values[i] = "3"
    
    data3 = dataTrain[dataTrain['dropout']== "3"]        
    dataTrain = dataTrain[dataTrain['dropout']!= "3"]
    dataTrain.dropout = dataTrain.dropout.astype(int)
    dataTrain2 = dataTrain.drop("nombre_carrera",1)
    #dataTrain1 = dataTrain1.drop("Unnamed: 0",1)
    	
    dataF = dataPred[dataPred.dropout == "2"]
    dataTest = dataPred[dataPred.dropout == "2"]
    dataTest['dropout'].replace("2","",inplace = True)
    dataTest = dataTest.drop("id",1).drop("nombre_carrera",1)
    for i in range(len(dataTest)):
        if dataTest['porcentaje_carrera'].values[i] >= 0.85: 
            dataTest['rateAprobadas'].values[i] = float(dataTest['rateAprobadas'].values[i])*1.2
            dataTest['mediaAPRP'].values[i] = float(dataTest['mediaAPRP'].values[i])*1.2
            dataTest['rateReprobadas'].values[i] = float(dataTest['rateReprobadas'].values[i])*0.8
            #dataTrain['mediaPond'].values[i] = dataTrain['mediaPond'].values[i]*1.05
            #dataTrain[''].values[i] = dataTrain['ratePass'].values[i]*1.05        
        elif dataTest['porcentaje_carrera'].values[i] >= 0.8: 
            dataTest['rateAprobadas'].values[i] = float(dataTest['rateAprobadas'].values[i])*1.17
            dataTest['mediaAPRP'].values[i] = float(dataTest['mediaAPRP'].values[i])*1.17
            dataTest['rateReprobadas'].values[i] = float(dataTest['rateReprobadas'].values[i])*0.83
            #dataTrain[''].values[i] = dataTrain['ratePass'].values[i]*1.05
        elif dataTest['porcentaje_carrera'].values[i] >= 0.7: 
            dataTest['rateAprobadas'].values[i] = float(dataTest['rateAprobadas'].values[i])*1.1
            dataTest['mediaAPRP'].values[i] = float(dataTest['mediaAPRP'].values[i])*1.1
            dataTest['rateReprobadas'].values[i] = float(dataTest['rateReprobadas'].values[i])*0.9
            #dataTrain['mediaPond'].values[i] = dataTrain['mediaPond'].values[i]*1.05
            #dataTrain[''].values[i] = dataTrain['ratePass'].values[i]*1.05
        elif dataTest['porcentaje_carrera'].values[i] >= 0.5: 
            dataTest['rateAprobadas'].values[i] = float(dataTest['rateAprobadas'].values[i])*1.05
            dataTest['mediaAPRP'].values[i] = float(dataTest['mediaAPRP'].values[i])*1.05            
            dataTest['rateReprobadas'].values[i] = float(dataTest['rateReprobadas'].values[i])*0.95
            #dataTrain['mediaPond'].values[i] = dataTrain['mediaPond'].values[i]*1.05
            #dataTrain[''].values[i] = dataTrain['ratePass'].values[i]*1.05

    """	
    x_train = dataTrain1.values[:,[8,9,10,11,12,13]]
    y_train = dataTrain1.values[:,18].astype(int)
    
    x_test1 = dataTest.values[:,[8,9,10,11,12,13]]
    """
    dataTrain1 = dataTrain[["rateReprobadas","rateAprobadas","mediaAPRP","mediaAP","segMatActual","terMatActual","terMatTot"]]#"numMatSem",
    x_train = dataTrain1.values[:,:]
    y_train = dataTrain['dropout'].astype(int).reset_index(drop = True)
    
    dataTest1 = dataTest[["rateReprobadas","rateAprobadas","mediaAPRP","mediaAP","segMatActual","terMatActual","terMatTot"]]#"numMatSem",
    x_test1 = dataTest1.values[:,:]
    
    robust_scaler = preprocessing.StandardScaler()
    x_train = robust_scaler.fit_transform(x_train)
    x_test = robust_scaler.fit_transform(x_test1)
    
    cfr = RandomForestClassifier()
    
    # Create the parameter grid based on the results of random search 
    """
    param_grid = {    
            'max_depth': range(2,20,4),
            'max_features': ['auto'],
            'min_samples_leaf': range(1,15,3),
            'min_samples_split': range(3,15,3),
            'n_estimators': range(300,400,100)
            }
    """
    param_grid = {    
            'max_depth': range(6,8,1),
            'max_features': ['auto'],
            'min_samples_leaf': range(7,8,4),
            'min_samples_split': range(6,7,3),
            'n_estimators': range(300,500,200),
            'random_state': range(10,11,1)
            }	
    
    # Instantiate the grid search model
    grid_search = GridSearchCV(estimator = cfr, param_grid = param_grid, cv = 10, n_jobs = -1, verbose = 2)
    # Fit the grid search to the data
    grid_search.fit(x_train, y_train)
    print("\n")
    print(grid_search.best_params_)
    print("\n")
    #Adaboost
    """
    param_ada = {
        'n_estimators': (150, 200),
        'base_estimator__max_depth': (1, 2),
        'algorithm': ('SAMME', 'SAMME.R')}
    ada = AdaBoostClassifier(base_estimator=DecisionTreeClassifier())
    grid_search = GridSearchCV(estimator = ada, param_grid = param_ada, cv = 10, n_jobs = -1, verbose = 2)
    # Fit the grid search to the data
    grid_search.fit(x_train, y_train)
    print("\n")
    print(grid_search.best_params_)
    print("\n")
    """
    #cfr2 = RandomForestClassifier(bootstrap = True, max_depth = 14, max_features='auto',min_samples_leaf=2,min_samples_split=8,n_estimators=475)
    #y_pred = cfr2.fit(x_train, y_train).predict_proba(x_test)
    #y_pr2 = cfr2.fit(x_train, y_train).predict(x_test)
    
    y_pred = grid_search.predict_proba(x_test)
    y_pr2 = grid_search.predict(x_test)
    
    
    #Bagging several RF
    y_pred_bag, y_pred_bag_c = ensemble.bagging(x_train, x_test, y_train, cfr, 10)
    #y_pred_bag.columns = ["dropout_bag"]
    
    #Averaging several models
    y_pred_avrg = ensemble.averaging_predictions(x_train, x_test, y_train)
    #y_pred_avrg.columns = ["dropout_avrg"]
    y_pred_stacking = ensemble.stacking_KFold_c(x_train, x_test, y_train, 3, 10)
    
    y_pred_calibrated = ensemble.model_calibration_prob(x_train, x_test, y_train, cfr)
   
    fig = plt.figure(figsize = (10, 8))
    sns.distplot(y_pred_bag, kde = True, label = "Bagging")    
    sns.distplot(y_pred, kde = True, label = "Grid_Search")       
    sns.distplot(y_pred_avrg, kde = True, label = "Average")          
    sns.distplot(y_pred_calibrated, kde = True, label = "Calibrated")    
    #sns.distplot(y_pred_stacking, kde = True, label = "Stacking")
    plt.legend()
    plt.title("Density Plot of dropout predictions" + carrera + str(i))
    
     
    x_pr1 = pd.DataFrame(x_test1)
    y_pr2 = pd.DataFrame(y_pr2)
    y_pred = pd.DataFrame(y_pred)
    y_pred.columns = ['X0', 'dropout']    
    y_pred = pd.DataFrame(y_pred['dropout'])
    y_pred_bag = pd.DataFrame(y_pred_bag)
    y_pred_bag.columns = ["dropout_bag"]
    y_pred_avrg = pd.DataFrame(y_pred_avrg)
    y_pred_avrg.columns = ["dropout_avrg"]
    y_pred_stacking = pd.DataFrame(y_pred_stacking)
    y_pred_stacking.columns = ["dropout_stack"]
    y_pred_calibrated = pd.DataFrame(y_pred_calibrated)
    y_pred_calibrated.columns = ["dropout_calibrated"]
    dataF = dataF.reset_index(drop=True)
    dataF.drop("dropout",1,inplace=True)

    df = pd.concat([dataF, y_pred, y_pred_bag, y_pred_avrg, y_pred_calibrated], axis=1)
    #df.columns = ['id','CLAS','stateStudent']
    df = df.drop("nombre_carrera",1)
    df[["rateReprobadas","rateAnuladas","rateAprobadas","mediaAPRP","mediaAP","segMatActual","terMatActual", "dropout", "dropout_bag", "dropout_avrg",  "dropout_calibrated"]].to_excel("Predicted.xlsx")
    #dataFin = pd.DataFrame(dataTrain1.values[:,[0,11]])
    dataFin = dataTrain2.copy()
    #dataFin.columns = ['id','stateStudent']
    data3.drop("nombre_carrera",1,inplace = True)
    finalData = pd.concat([dataFin, df, data3],axis = 0)
    #finalData.to_excel("Results.xlsx")
    finalData['mediaCarreraGraduadosAP'] = mediaCarreraAPGrad['mediaAP'].values[0]
    finalData['mediaCarreraAbandonadosAP'] = mediaCarreraAPAbandon['mediaAP'].values[0]
    finalData['mediaCarreraGraduadosAPRP'] = mediaCarreraAPRPGrad['mediaAPRP'].values[0]
    finalData['mediaCarreraAbandonadosAPRP'] = mediaCarreraAPRPAbandon['mediaAPRP'].values[0]    
    finalData['mediaCarreraGraduadosRP'] = mediaCarreraRPGrad['mediaRP'].values[0]
    finalData['mediaCarreraAbandonadosRP'] = mediaCarreraRPAbandon['mediaRP'].values[0]
    finalData['mediaCarreraGraduadosSecRP'] = mediaCarreraSecRPGrad['terMatTot'].values[0]
    finalData['mediaCarreraAbandonadosSecRP'] = mediaCarreraSecRPAbandon['terMatTot'].values[0]
    finalData['rateCarreraGraduadosRP'] = mediaCarreraRateRPGrad['rateReprobadas'].values[0]
    finalData['rateCarreraAbandonadosRP'] = mediaCarreraRateRPAbandon['rateReprobadas'].values[0]
    finalData['rateCarreraGraduadosAP'] = mediaCarreraRateAPGrad['rateAprobadas'].values[0]
    finalData['rateCarreraAbandonadosAP'] = mediaCarreraRateAPAbandon['rateAprobadas'].values[0]
    
    finalData = finalData.sort_values(by='id', ascending=True)
    #dataPred.drop("stateStudent",1, inplace = True)
    #finalData = pd.merge(dataPred, finalData, how='outer', on=['id'])
    """
    finalData.to_excel("Results.xlsx")
    finalData = finalData.drop("id",1)
    """
    return finalData