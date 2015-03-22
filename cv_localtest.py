import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
from sklearn.tree import DecisionTreeRegressor
from sklearn import ensemble
from sklearn import cross_validation
from datetime import datetime
import matplotlib.pyplot as plt

def main():
    
    #gather data from train and test csv 
    data_train = pd.read_csv('train.csv', skipinitialspace=1, index_col=0, parse_dates=True)
    
    #explode datetime into hour, dayofweek, and month
    data_train['hour'] = data_train.index.hour
    data_train['dayofweek'] = data_train.index.weekday
    data_train['month'] = data_train.index.month
        
    #build two data sets: one for registered users and one for casual, splitting data here as appropriate
    X_selected_cols = [u'weather',u'month',u'dayofweek',u'hour',u'season',u'holiday',u'workingday',u'temp',u'atemp',u'humidity',u'windspeed']
    X_train = data_train[X_selected_cols]
    y_train_reg = np.log(pd.Series(data_train['registered'], dtype='float'))
    y_train_reg[y_train_reg < 0] = 0
    y_train_cas = np.log(pd.Series(data_train['casual'], dtype='float'))
    y_train_cas[y_train_cas < 0] = 0

    #create models
    clf_reg = DecisionTreeRegressor(max_depth=9)
    clf_cas = DecisionTreeRegressor(max_depth=9)
    
    #compute scores, print accuracy
    scores_reg = cross_validation.cross_val_score(clf_reg, X_train, y_train_reg, 'mean_squared_error')
    scores_cas = cross_validation.cross_val_score(clf_cas, X_train, y_train_cas, 'mean_squared_error')
    print "Accuracy: %0.2f " % ((-scores_reg.mean() + -scores_cas.mean()) / 2)

if __name__=="__main__":
    main()