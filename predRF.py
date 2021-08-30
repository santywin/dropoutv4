import numpy as np
import pandas as pd 
from sklearn import model_selection
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
import matplotlib.pyplot as plt
from sklearn import metrics

#def main():
def predictStat(final,carrera):
    #file = 'Final.csv'
    #data = pd.read_csv(file)
    data = final.copy()
    #data.nombre_carrera = data.to_string(columns = ['nombre_carrera'])
    dataPred = data[data.nombre_carrera == carrera]
    #dataPred = data[data.nombre_carrera == "INGENIERIA CIVIL-REDISEÑO"]
    dataPred = dataPred.drop("id",1)
    #dataPred['DROPOUT'] = dataPred['DROPOUT'].str.strip()
    
        
    dataPred['dropout'].fillna("2",inplace = True)
    #print(dataPred.stateStudent)
    dataPred = dataPred.dropna()
    #print(len(dataPred))    
    dataPred['dropout'].replace("","2",inplace = True)
    dataPred = dataPred.fillna(0)
    
    dataTrain = dataPred[dataPred.dropout != "2"]
    
    mediaCarrera = dataTrain.groupby(['nombre_carrera']).mediaAP.sum().to_frame()/dataTrain.groupby(['nombre_carrera']).mediaAP.count().to_frame()
    mediaCarreraRateP = dataTrain.groupby(['nombre_carrera']).rateAprobadas.sum().to_frame()/dataTrain.groupby(['nombre_carrera']).rateAprobadas.count().to_frame()
    
    aprobados = dataTrain[dataTrain['dropout'] == "0"]
    abandonados = dataTrain[dataTrain['dropout'] == "1"]
    abandonados = abandonados[abandonados['mediaAP'] != 0]
    mediaCarreraAPGrad = aprobados.groupby(['nombre_carrera']).mediaAP.sum().to_frame()/aprobados.groupby(['nombre_carrera']).mediaAP.count().to_frame()
    mediaCarreraAPAbandon = abandonados.groupby(['nombre_carrera']).mediaAP.sum().to_frame()/abandonados.groupby(['nombre_carrera']).mediaAP.count().to_frame()
    mediaCarreraAPRPGrad = aprobados.groupby(['nombre_carrera']).mediaAPRP.sum().to_frame()/aprobados.groupby(['nombre_carrera']).mediaAPRP.count().to_frame()
    mediaCarreraAPRPAbandon = abandonados.groupby(['nombre_carrera']).mediaAPRP.sum().to_frame()/abandonados.groupby(['nombre_carrera']).mediaAPRP.count().to_frame()
    mediaCarreraRPGrad = aprobados.groupby(['nombre_carrera']).mediaRP.sum().to_frame()/aprobados.groupby(['nombre_carrera']).mediaRP.count().to_frame()
    mediaCarreraRPAbandon = abandonados.groupby(['nombre_carrera']).mediaRP.sum().to_frame()/abandonados.groupby(['nombre_carrera']).mediaRP.count().to_frame()
    mediaCarreraSecRPGrad = aprobados.groupby(['nombre_carrera']).segMatTot.sum().to_frame()/aprobados.groupby(['nombre_carrera']).segMatTot.count().to_frame()
    mediaCarreraSecRPAbandon = abandonados.groupby(['nombre_carrera']).segMatTot.sum().to_frame()/abandonados.groupby(['nombre_carrera']).segMatTot.count().to_frame()
    
    #Media rates
    mediaCarreraRateRPGrad = aprobados.groupby(['nombre_carrera']).rateReprobadas.sum().to_frame()/aprobados.groupby(['nombre_carrera']).rateReprobadas.count().to_frame()
    mediaCarreraRateRPAbandon = abandonados.groupby(['nombre_carrera']).rateReprobadas.sum().to_frame()/abandonados.groupby(['nombre_carrera']).rateReprobadas.count().to_frame()
    mediaCarreraRateAPGrad = aprobados.groupby(['nombre_carrera']).rateAprobadas.sum().to_frame()/aprobados.groupby(['nombre_carrera']).rateAprobadas.count().to_frame()
    mediaCarreraRateAPAbandon = abandonados.groupby(['nombre_carrera']).rateAprobadas.sum().to_frame()/abandonados.groupby(['nombre_carrera']).rateAprobadas.count().to_frame()
    #dataTrain['mediaCarreraAP'] = mediaCarrera['mediaAP'].values[0]
        
    for i in range(len(dataTrain)):
        if  dataTrain['rateAprobadas'].values[i] >= mediaCarreraRateP['rateAprobadas'].values[0]*1.05 and dataTrain['dropout'].values[i] == "1" and dataTrain['mediaAP'].values[i] >= mediaCarrera['mediaAP'].values[0]*1.05:
            dataTrain['dropout'].values[i] = "3"
    
    data3 = dataTrain[dataTrain['dropout']== "3"]        
    dataTrain = dataTrain[dataTrain['dropout']!= "3"]
    dataTrain.dropout = dataTrain.dropout.astype(int)
    dataTrain = dataTrain.drop("nombre_carrera",1)
    
    dataTest = dataPred[dataPred.dropout == "2"]
    dataTest['dropout'].replace("2","",inplace = True)
    
    correlation = dataTrain[["dropout","rateReprobadas","rateAnuladas","rateAprobadas","mediaRP","mediaAPRP","mediaAP","numMatSem","segMatActual","segMatTot","terMatActual","terMatTot"]].corr().fillna(0)
    #dataTrain.to_excel("dataTrain.xlsx")
    #dataTest.to_excel("dataTest.xlsx")
    dataTrain1 = dataTrain[["rateReprobadas","rateAnuladas","rateAprobadas","mediaAPRP","mediaAP","segMatActual","terMatActual","terMatTot"]]#"numMatSem",
    x_train = dataTrain1.values[:,:]
    y_train = dataTrain['dropout'].astype(int).reset_index(drop = True)
    
    #x_test = dataTest.values[:,[3,5,6,7,8,9]]
    #y_test = dataTest.values[:,13]
    
    robust_scaler = preprocessing.StandardScaler()
    x_train = robust_scaler.fit_transform(x_train)
    #x_test = robust_scaler.fit_transform(x_test)
    print("Starting cross-validation (" +  str(len(x_train)) + ' learners)')
    
    #cfr = RandomForestRegressor(n_estimators = 500)
    cfr2 = RandomForestClassifier(n_estimators = 500)
        
    kf = model_selection.KFold(n_splits=10)
    cv = kf.split(x_train)
        
    results = []
    res_ce = []
    A_A = 0
    A_S = 0
    S_A = 0
    S_S = 0
    
    y_pred_list = list()
    y_true_list = list()
    y_true2_list = list()
    
    for traincv, testcv in cv:
        #y_pred = cfr.fit(x_train[traincv], y_train[traincv]).predict_proba(x_train[testcv])
        #results.append(np.sqrt(np.mean((y_pred - y_train[testcv])**2)))
        y_pred1 = cfr2.fit(x_train[traincv], y_train[traincv])
        y_pred = y_pred1.predict_proba(x_train[testcv])
        #results.append(np.sqrt(np.mean((y_pred[:,1] - y_train[testcv])**2)))
        
        y_pr2 = cfr2.fit(x_train[traincv], y_train[traincv]).predict(x_train[testcv])
        res_ce.append(np.mean(np.abs(y_pr2 - y_train[testcv])))
        y_pr1 = pd.DataFrame(y_train[testcv])
        x_pr1 = pd.DataFrame(x_train[testcv])
        y_pred2 = pd.DataFrame(y_pr2)
        if len(y_pred[0,:]) == 1:
            y_pred = pd.DataFrame(y_pred)
            y_pred[1] = 1 - y_pred[0]
            y_pred = np.array(y_pred)
        df = pd.DataFrame(y_pred[:,1] )
        df = pd.concat([y_pr1,y_pred2, df], axis=1,)
        #print(df)
        #df.to_excel("ResultsRF.xlsx")
    
        # Store results for AUC
        for i, v in enumerate(y_pred[:,1]):
            y_pred_list.append(v)
             #print(y_train[testcv])
            y_true_list.append(y_train[testcv].iloc[i])
            #print(y_train[testcv].iloc[i])
            y_true2_list.append(y_train[testcv].iloc[i])
            # Certificate earners
        for i, val in enumerate(y_pr2):
            if y_pr2[i] == 1 and y_train[testcv].iloc[i] == 1:
                A_A += 1
            if y_pr2[i] == 0 and  y_train[testcv].iloc[i] == 1:
                A_S += 1
            if y_pr2[i] == 1  and  y_train[testcv].iloc[i] == 0:
                S_A += 1
            if y_pr2[i] == 0  and  y_train[testcv].iloc[i] == 0:
                S_S += 1
    #print out the mean of the cross-validated results
    #RMSE = np.array(results).mean()
    #print("RMSE: " + str( RMSE))
    accuracy = (A_A+S_S)/((A_A+A_S+S_A+S_S)*1.0)
    print("Results CE: " + str(1-np.array(res_ce).mean()) + " / " + str(accuracy))
    # Results about certificate earners11 10, 01 00
    print(str(A_A) + "\t" + str(A_S))
    print(str(S_A) + "\t" + str(S_S))
    TP = A_A
    FP = A_S
    FN = S_A
    TN = S_S
    try:
        recall = TP / ((TP+FN)*1.0);
    except:
        recall = 0
    try:
        precision = TP / ((TP+FP)*1.0);
    except:
        precision = 0
    try:
        specificity = TN / ((TN+FP)*1.0)
    except:
        specificity = 0
    try:
        NPV = TN / ((FN+TN)*1.0);
    except:
        NPV = 0
    try:
        F_score = (2*TP)/((2*TP+FP+FN)*1.0)
    except:
        F_score = 0
    
    print('Recall: ' + str(recall))
    print('Precision: ' + str(precision))
    print('Specificity: ' + str(specificity))
    print('NVP:' + str(NPV))
    print('F-score: ' + str(F_score))
    
    # Compute AUC
    y = np.array(y_true_list)
    pred = np.array(y_pred_list)
    y_true = np.array(y_true2_list)
    fpr, tpr, thresholds = metrics.roc_curve(y, pred)
    
    AUC = metrics.auc(fpr, tpr)
    RMSEsk = np.sqrt(metrics.mean_squared_error(y_true, pred))
    MAE = metrics.mean_absolute_error(y_true, pred)
    print('AUC: ' + str(AUC))
    
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
                lw=lw, label='ROC curve (area = %0.2f)' % metrics.auc(fpr, tpr))
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()
    
    results = dict()
    results['RMSE'] = RMSEsk
    results['MAE'] = MAE
    results['AUC'] = AUC
    results['F1'] = F_score
    results['recall'] = recall
    results['precision'] = precision
    results['accuracy'] = accuracy
    
    print(results)
    result = pd.DataFrame([[key, results[key]] for key in results.keys()],columns = ['metric','amount'])
    #resultados['amount'] = result['amount'].values[:]
    return result['amount'].values[:], result['metric'].values[:],correlation
    
    
    
        
    
    #if __name__ == "__main__":
    #           main()