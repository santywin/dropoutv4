#!/usr/bin/env python
# coding: utf-8
import numpy as np


def bagging(train, test, y, model, bags):
    seed = 1
    bagged_prediction = np.zeros(test.shape[0])
    bagged_prediction_c = np.zeros(test.shape[0])
    for n in range(0, bags):
        model.set_params(random_state = seed + n)
        model.fit(train, y)
        pred_c = model.predict(test)
        pred = model.predict_proba(test)
        bagged_prediction += pred[:,1]
        bagged_prediction_c += pred_c 
    #Average
    bagged_prediction /= bags
    bagged_prediction_c /= bags
    bagged_prediction_c = (np.around(bagged_prediction_c))
    return bagged_prediction, bagged_prediction_c

#model = RandomForestRegressor(n_estimators = 150, max_depth = 7, n_jobs = -1)


# In[ ]:

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import AdaBoostClassifier
import pandas as pd

def averaging_predictions(x_train, x_test, y_train):
    pred_ada = pd.DataFrame(AdaBoostClassifier(n_estimators = 150).fit(x_train, y_train).predict_proba(x_test)[:,1])
    pred_rf = pd.DataFrame(RandomForestClassifier(n_estimators = 150).fit(x_train, y_train).predict_proba(x_test)[:,1])
    pred_et = pd.DataFrame(ExtraTreesClassifier(n_estimators = 150).fit(x_train, y_train).predict_proba(x_test)[:,1])
    
    predictions = [pred_ada, pred_rf, pred_et]    
    X_test_level2 = pd.concat(predictions, axis = 1) 
    pred_average = X_test_level2.sum(axis = 1)/len(predictions)
    return pred_average
    
    
#predictions = [pred_lr, pred_lgb, pred_rf, pred_svr_rbf]


# In[ ]:

from sklearn import svm
from sklearn import linear_model
#Stacking example
def stacking_timeseries(x_train, x_test, y_train, nalgorithms):
    # That is how we get target for the 2nd level dataset
    y_train_level2 = y_train.copy()
    X_train_level2 = np.zeros([y_train_level2.shape[0], nalgorithms])
    #Cross-validation
    for cur_block_num in range(12,max(dates_train_level2)):
    
        print(cur_block_num)

        '''
            1. Split `X_train` into parts
               Remember, that corresponding dates are stored in `dates_train` 
            2. Fit linear regression 
            3. Fit LightGBM and put predictions          
            4. Store predictions from 2. and 3. in the right place of `X_train_level2`. 
               You can use `dates_train_level2` for it
               Make sure the order of the meta-features is the same as in `X_test_level2`
        '''      

        X_train_f = X_train.loc[dates_train<= cur_block_num]
        y_train_f = y_train[dates_train<= cur_block_num]

        X_test_f = X_train.loc[dates_train == cur_block_num]

        lr.fit(X_train_f.values, y_train_f)
        pred_lr_fold = lr.predict(X_test_f.values)

        model = lgb.train(lgb_params, lgb.Dataset(X_train_f, label=y_train_f), 100)
        pred_lgb_fold = model.predict(X_test_f)

        #model = regr.fit(X_train_f, y_train_f)
        #pred_rf_fold = model.predict(X_test_f)

        model = svr_rbf.fit(X_train_f, y_train_f)
        pred_svr_fold = model.predict(X_test_f)

        X_train_level2[dates_train_level2 == cur_block_num] = np.c_[pred_lr_fold, pred_lgb_fold, pred_svr_fold]

    X_train_level2_save = pd.DataFrame(X_train_level2)
    stack_lr = LinearRegression()
    stack_lr.fit(X_train_level2, y_train_level2)
    train_preds = stack_lr.predict(X_test_level2)


# In[ ]:
from sklearn.model_selection import KFold
def stacking_KFold_c(x_train, x_test, y_train, nalgorithms, nfolds):
    # That is how we get target for the 2nd level dataset
    y_train_level2 = y_train.copy()
    X_train_level2 = np.zeros([y_train_level2.shape[0], nalgorithms])
    #Cross-validation
    
    kfold = KFold(n_splits = nfolds, shuffle = False, random_state = 1)

    for train, test in kfold.split(x_train):
    
        

        '''
            1. Split `X_train` into parts
               Remember, that corresponding dates are stored in `dates_train` 
            2. Fit linear regression 
            3. Fit LightGBM and put predictions          
            4. Store predictions from 2. and 3. in the right place of `X_train_level2`. 
               You can use `dates_train_level2` for it
               Make sure the order of the meta-features is the same as in `X_test_level2`
        '''      

        X_train_f = x_train[train]
        y_train_f = y_train[train]

        X_test_f = x_train[test]
        
        pred_rf_fold = pd.DataFrame(RandomForestClassifier(n_estimators = 150).fit(X_train_f, y_train_f).predict_proba(X_test_f)[:,1])
        pred_svm_fold = pd.DataFrame(svm.SVC(kernel = 'rbf', probability = True).fit(X_train_f, y_train_f).predict_proba(X_test_f)[:,1])
        pred_linear = pd.DataFrame(linear_model.SGDClassifier(loss = 'log', max_iter=1000, tol=1e-3).fit(X_train_f, y_train_f).predict_proba(X_test_f)[:,1])
        
        X_train_level2[test] = pd.concat([pred_rf_fold, pred_svm_fold, pred_linear], axis = 1)
    #Create test set level 2
    pred_rf_fold = pd.DataFrame(RandomForestClassifier(n_estimators = 150).fit(x_train, y_train).predict_proba(x_test)[:,1])
    pred_svm_fold = pd.DataFrame(svm.SVC(kernel = 'rbf', probability = True).fit(x_train, y_train).predict_proba(x_test)[:,1])
    pred_linear = pd.DataFrame(linear_model.SGDClassifier(loss = 'log', max_iter=1000, tol=1e-3).fit(x_train, y_train).predict_proba(x_test)[:,1])
        
    X_test_level2 = pd.concat([pred_rf_fold, pred_svm_fold, pred_linear], axis = 1)

    X_train_level2_save = pd.DataFrame(X_train_level2)
    #stack_RF = linear_model.LogisticRegression()
    stack_RF = RandomForestClassifier(n_estimators = 150, max_depth = 7)
    stack_RF.fit(X_train_level2, y_train_level2)
    test_preds = stack_RF.predict_proba(X_test_level2)[:,1]
    return test_preds

from sklearn.calibration import calibration_curve
from sklearn.calibration import CalibratedClassifierCV

def model_calibration_prob(x_train, x_test, y_train, model):
    calibrated = CalibratedClassifierCV(model, method='sigmoid', cv=5)
    calibrated.fit(x_train, y_train)
    # predict probabilities
    probs = calibrated.predict_proba(x_test)[:, 1]
    """
    # reliability diagram
    fop, mpv = calibration_curve(testy, probs, n_bins=10, normalize=True)
    # plot perfectly calibrated
    pyplot.plot([0, 1], [0, 1], linestyle='--')
    # plot calibrated reliability
    pyplot.plot(mpv, fop, marker='.')
    pyplot.show()
    """
    return probs
