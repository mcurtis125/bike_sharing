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
    #y_train_firsthalf= data_train[((data_train['weekofyear'] %2 == 0))]
    y_train_evenweek= data_train[((data_train['weekofyear'] %2 == 0))]
    #y_train_evenweek[y_train_reg %2 == 0] = 0
    y_test_oddweek= data_train[((data_train['weekofyear'] %2 != 0))]
   # y_test_oddweek[y_train_reg % 2 != 0] = 0

    y_train_evenweek['count'] = pd.DataFrame(y_train_evenweek, dtype='int', columns=['count'])
    y_test_oddweek['count'] = pd.DataFrame(y_test_oddweek, dtype='int', columns=['count'])

    #compute error by taking log square
    count_error = np.power(np.log(y_train_evenweek['count']+1) - np.log(y_test_oddweek['count']+1),2)
    print count_error.head(10)
    print  y_train_evenweek['count'].head(10)
    print y_test_oddweek['count'].head(10)

    # even_count = y_train_evenweek.groupby('weekofyear').agg({'count': [np.mean, np.std] })
    # even_count.columns = even_count.columns.droplevel(0)
    # plt.figure().canvas.set_window_title('Bike Count in even weeks')
    # bar_width = 0.5
    # plt.bar(y_train_evenweek.index , even_count['mean'], bar_width, alpha=0.4, color='b', ecolor='r', label='Count')
    # plt.ylabel('Mean Count')
    # plt.xlabel('Even Weeks')
    # plt.title ('Bike Count in even weeks')
    # plt.show()
if __name__=="__main__":
    main()