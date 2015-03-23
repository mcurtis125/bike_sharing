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
    
    plt.subplot(2, 1, 1)
    plt.plot(data_train.index, data_train['count'], '-yo')
    plt.title('Training Data: Datetime VS. Count')
    plt.ylabel('Count')
    plt.xlabel('Datetime')
    
    plt.subplot(2, 1, 2)
    plt.plot(data_test.index, data_test['count'], '-yo')
    plt.title('Training Data: Datetime VS. Count')
    plt.ylabel('Count')
    plt.xlabel('Datetime')
    
    plt.show()
    
    #build two data sets: one for registered users and one for casual, splitting data here as appropriate
    X_selected_cols = [u'weather',u'month',u'dayofweek',u'hour',u'season',u'holiday',u'workingday',u'temp',u'atemp',u'humidity',u'windspeed']
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
	
	#plot squared log error by day
    error = np.power( (np.log(np.array(data_test['count']) + 1) - np.log(np.array(y_test['count']) + 1)), 2 )
    plt.subplot(2, 1, 1)
    plt.plot(y_test['datetime'], error, '-yo')
    plt.title('Error vs time')
    plt.ylabel('Squared Log Error')
    plt.xlabel('Datetime')
    
    #plot number of errors by time of day (# errors above a threshold)
    plt.subplot(2, 1, 2)
    error_hour = np.zeros(24)
    for i in range(0,error.size-1):
        if error[i] > 2:
            error_hour[data_test['hour'][i]] = error_hour[data_test['hour'][i]]+1
    x = np.arange(0, 24, 1)
    bar_width = 1
    plt.bar(x, error_hour, 1, color='y', ecolor='y', label='#Errors > 0.5')
    plt.xticks(x)
    plt.title('Error vs hour of the day')
    plt.ylabel('# Errors > 2')
    plt.xlabel('Hour of the day')

    plt.show()
	
	#print root mean squared log error
    print np.sqrt(np.sum(error)/error.size)
    
    #printing to csv
    y_test.to_csv('simple_output.csv', columns=['datetime', 'count'], index=0)
	

if __name__=="__main__":
    main()