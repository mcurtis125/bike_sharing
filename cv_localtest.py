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
    data_train['weekofyear'] =data_train.index.weekofyear

        
    #build two data sets: one for registered users and one for casual, splitting data here as appropriate
    X_selected_cols = [u'weather',u'month',u'dayofweek',u'hour',u'season',u'holiday',u'workingday',u'temp',u'atemp',u'humidity',u'windspeed']
    X_train = data_train[X_selected_cols]
    y_train_reg = np.log(pd.Series(data_train['registered'], dtype='float'))
    y_train_reg[y_train_reg < 0] = 0
    y_train_cas = np.log(pd.Series(data_train['casual'], dtype='float'))
    y_train_cas[y_train_cas < 0] = 0

    #build two data sets: one for the first half year and one for the second half year
    #y_train_firsthalf= data_train[(data_train['weekofyear'] <26)]
    y_train_evenweek= np.log(pd.Series(data_train['weekofyear'], dtype='int'))
    y_train_evenweek[y_train_reg %2 == 0] = 0
    y_test_oddweek= np.log(pd.Series(data_train['weekofyear'], dtype='int'))
    y_test_oddweek[y_train_reg % 2 != 0] = 0


    #create models
    clf_reg = DecisionTreeRegressor(max_depth=9)
    clf_cas = DecisionTreeRegressor(max_depth=9)
<<<<<<< HEAD

    # #compute scores, print accuracy
    # scores_reg = cross_validation.cross_val_score(clf_reg, X_train, y_train_evenweek, 'mean_squared_error')
    # scores_cas = cross_validation.cross_val_score(clf_cas, X_train, y_test_oddweek, 'mean_squared_error')
    # print "Accuracy: %0.2f " % ((-scores_reg.mean() + -scores_cas.mean()) / 2)

    #computer score using log square
    scores_evenweek = evaluation(X_train, y_train_evenweek);
    scores_oddweek = evaluation(X_train, y_train_oddweek);
    print "Accuracy: ((scores_evenweek.mean() + scores_oddweek.mean()) / 2)"
=======
    
    #compute scores, print accuracy
    scores_reg = cross_validation.cross_val_score(clf_reg, X_train, y_train_reg, 'mean_squared_error')
    scores_cas = cross_validation.cross_val_score(clf_cas, X_train, y_train_cas, 'mean_squared_error')
    print "Accuracy: %0.2f " % ((-scores_reg.mean() + -scores_cas.mean()) / 2)
>>>>>>> a09a61faa1361dbe0e60918ec71d76f915066268

#method to print our kaggle score if we were to submit this algorithm
def evaluation(predicted, actual):
    sum = 0
    for row_index in range(0, len(actual) - 1):
        pi = float(predicted[row_index][0])
        ai = float(actual[row_index][0])
        sum += log_sq_diff(pi, ai)
    print "Evaluation Score = %f" % ( math.sqrt( (1/float(len(predicted))) * sum ) )
#method to find the log squared diffence between predicted and actual
def log_sq_diff(pi, ai):
    return math.pow( (math.log(pi + 1) - math.log(ai + 1)), 2 )
 
if __name__=="__main__":
    main()