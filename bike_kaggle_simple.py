import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
from sklearn import ensemble
from datetime import datetime

def main():
    
    #gather data from train and test csv 
    data_train = pd.read_csv('train.csv', skipinitialspace=1, index_col=0, parse_dates=True)
    data_test = pd.read_csv('test.csv', skipinitialspace=1, index_col=0, parse_dates=True)
    
    #explode datetime into hour, dayofweek, and month
    data_train['hour'] = data_train.index.hour
    data_train['dayofweek'] = data_train.index.weekday
    data_train['month'] = data_train.index.month
    data_test['datetime'] = data_test.index
    data_test['hour'] = data_test.index.hour
    data_test['dayofweek'] = data_test.index.weekday
    data_test['month'] = data_test.index.month
                
    #build two data sets: one for registered users and one for casual, splitting data here as appropriate
    X_selected_cols = [u'weather',u'month',u'dayofweek',u'hour']
    X_train = data_train[X_selected_cols]
    X_test = data_test[X_selected_cols]
    y_train_reg = np.log(pd.Series(data_train['registered'], dtype='float'))
    y_train_reg[y_train_reg < 0] = 0
    y_train_cas = np.log(pd.Series(data_train['casual'], dtype='float'))
    y_train_cas[y_train_cas < 0] = 0
    
    #creating the registered and casual models for ride share demand
    params = {'n_estimators': 1000, 'max_depth': 3, 'min_samples_split': 1,'learning_rate': 0.01, 'loss': 'ls'}
    clf_reg = ensemble.GradientBoostingRegressor(**params)
    clf_cas = ensemble.GradientBoostingRegressor(**params)
    
    #find the fit for both models
    clf_reg.fit(X_train, y_train_reg)
    clf_cas.fit(X_train, y_train_cas)
    
    #predict the registered and casual users with the test data
    y_test_reg = clf_reg.predict(X_test)
    y_test_cas = clf_cas.predict(X_test)
    
    #take the exponential of the outputs to account for the log of the original input
    y_test_reg = np.exp(pd.Series(y_test_reg, dtype='float'))
    y_test_cas = np.exp(pd.Series(y_test_cas, dtype='float'))
    
    #add together to find predicted count
    y_test = y_test_reg + y_test_cas
    
    #groom for csv printing
    y_test_datetime = pd.DataFrame(data_test.index, columns=['datetime'])
    y_test_count = pd.DataFrame(y_test, dtype='float', columns=['count'])
    y_test = pd.DataFrame()
    y_test['datetime'] = y_test_datetime['datetime']
    y_test['count'] = y_test_count['count']
    
    #printing to csv
    y_test.to_csv('simple_output.csv', columns=['datetime', 'count'], index=0)

if __name__=="__main__":
    main()