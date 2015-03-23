import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
from sklearn import ensemble, cross_validation 
from sklearn.metrics import mean_squared_error
from datetime import datetime
import matplotlib.pyplot as plt

def main():
    
    #gather data from train and test csv 
    data = pd.read_csv('train.csv', skipinitialspace=1, index_col=0, parse_dates=True)
    
    #explode datetime into hour, dayofweek, and month
    data['hour'] = data.index.hour
    data['dayofweek'] = data.index.weekday
    data['month'] = data.index.month
    data['weekofyear'] = data.index.weekofyear
    
    #split the training data into a validation split
    data_train = data[data['weekofyear'] % 2 == 0]
    data_test = data[data['weekofyear'] % 2 != 0]
    
    #build two data sets: one for registered users and one for casual, splitting data here as appropriate
    X_selected_cols = [u'weather',u'month',u'dayofweek',u'hour',u'season',u'holiday',u'workingday',u'temp',u'atemp',u'humidity',u'windspeed']
    X_train = data_train[X_selected_cols]
    X_test = data_test[X_selected_cols]
    y_train_reg = np.log(pd.Series(data_train['registered'], dtype='float'))
    y_train_reg[y_train_reg < 0] = 0
    y_train_cas = np.log(pd.Series(data_train['casual'], dtype='float'))
    y_train_cas[y_train_cas < 0] = 0
    
    #creating the registered and casual models for ride share demand
    params = {'n_estimators': 1000, 'max_depth': 5, 'min_samples_split': 1,'learning_rate': 0.01, 'loss': 'ls'}
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
    y_test_weather = pd.DataFrame(data_test['weather'], dtype='int', columns=['weather'])
    y_test = pd.DataFrame()
    y_test['datetime'] = y_test_datetime['datetime']
    y_test['count'] = y_test_count['count']
    y_test['weather'] = y_test_weather['weather'].astype(int)
	
    #find the error for graphing
    y_test['logerror'] = pd.Series(np.absolute(np.log(np.array(y_test['count'])) - np.log(np.array(data_test['count']))))
    y_test = y_test.set_index('datetime')
    y_test['hour'] = y_test.index.hour
    y_test['dayofweek'] = y_test.index.weekday
    
    plt.plot(y_test.index, y_test['logerror'], '-yo')
    plt.title('Error vs time')
    plt.ylabel('Squared Log Error')
    plt.xlabel('Datetime')
    plt.show()
    
    #plot by hour average count for each hour in the day on weekend and and weekdays
    hour_workday = y_test[(y_test['dayofweek'] < 5)].groupby('hour').agg({'logerror': [np.mean, np.std, np.count_nonzero]})
    hour_weekend = y_test[(y_test['dayofweek'] > 4)].groupby('hour').agg({'logerror': [np.mean, np.std, np.count_nonzero]})
    #droplevel statement just allows access to the mean and std columns
    hour_workday.columns = hour_workday.columns.droplevel(0)
    hour_weekend.columns = hour_weekend.columns.droplevel(0)
    plt.figure().canvas.set_window_title('Training Data: Hour in Day VS. Mean Error on Hour')
    #plot each bar
    bar_width = 0.4
    plt.bar(hour_workday.index, hour_workday['mean'], bar_width, yerr=hour_workday['std'], alpha=0.4, color='b', ecolor='b', label='Workday')
    plt.bar(hour_weekend.index + bar_width, hour_weekend['mean'], bar_width, yerr=hour_weekend['std'], alpha=0.4, color='g', ecolor='g', label='Weekend')
    #chart labeling
    plt.xticks(hour_workday.index + bar_width, hour_workday.index)
    plt.ylabel('Mean Error: (log(predicted) - log(actual))^2')
    plt.xlabel('Hour in Day')
    plt.title('Training Data: Hour in Day VS. Mean Error on Hour')
    plt.legend()
    plt.show()

'''
    #plot by hour average count for each hour in the day on weekend and and weekdays
    hour_workday = y_test[(y_test['dayofweek'] < 5)].groupby('weather').agg({'logerror': [np.mean, np.std]})
    hour_weekend = y_test[(y_test['dayofweek'] > 4)].groupby('weather').agg({'logerror': [np.mean, np.std]})
    #droplevel statement just allows access to the mean and std columns
    hour_workday.columns = hour_workday.columns.droplevel(0)
    hour_weekend.columns = hour_weekend.columns.droplevel(0)
    plt.figure().canvas.set_window_title('Training Data: Weather Type VS. Mean Error on Hour')
    #plot each bar
    bar_width = 0.4
    plt.bar(hour_workday.index, hour_workday['mean'], bar_width, yerr=hour_workday['std'], alpha=0.4, color='b', ecolor='b', label='Workday')
    plt.bar(hour_weekend.index + bar_width, hour_weekend['mean'], bar_width, yerr=hour_weekend['std'], alpha=0.4, color='g', ecolor='g', label='Weekend')
    #chart labeling
    plt.xticks(hour_workday.index + bar_width, hour_workday.index)
    plt.ylabel('Mean Error: (log(predicted) - log(actual))^2')
    plt.xlabel('Weather Type')
    plt.title('Training Data: Weather Type VS. Mean Error on Hour')
    plt.legend()
    plt.show()
'''

if __name__=="__main__":
    main()