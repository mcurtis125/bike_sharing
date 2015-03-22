import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
from sklearn import ensemble, cross_validation
from datetime import datetime
import matplotlib.pyplot as plt

def main():
    
    #gather data from train and test csv 
    data_train = pd.read_csv('train.csv', skipinitialspace=1, index_col=0, parse_dates=True)
    data_test = pd.read_csv('test.csv', skipinitialspace=1, index_col=0, parse_dates=True)
    
    #explode datetime into hour, dayofweek, and month
    data_train['hour'] = data_train.index.hour
    data_train['dayofweek'] = data_train.index.weekday
    data_train['month'] = data_train.index.month
    data_test['hour'] = data_test.index.hour
    data_test['dayofweek'] = data_test.index.weekday
    data_test['month'] = data_test.index.month
    
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
    
    #printing to csv
    y_test.to_csv('simple_output.csv', columns=['datetime', 'count'], index=0)
    
    #plot datetime vs count
    plt.plot(data_train.index, data_train['count'], '-yo')
    plt.title('Training Data: Datetime VS. Count')
    plt.ylabel('Count')
    plt.xlabel('Datetime')
    plt.show()
    
    #plot day of week vs mean count on day for each count type
    data_train_night = data_train[((data_train['hour'] > 23) | (data_train['hour'] < 6))]
    day_count = data_train_night.groupby('dayofweek').agg({'count': [np.mean, np.std] })
    day_reg = data_train_night.groupby('dayofweek').agg({'registered': [np.mean, np.std] })
    day_cas = data_train_night.groupby('dayofweek').agg({'casual': [np.mean, np.std] })
    #droplevel statement just allows access to the mean and std columns
    day_count.columns = day_count.columns.droplevel(0)
    day_reg.columns = day_reg.columns.droplevel(0)
    day_cas.columns = day_cas.columns.droplevel(0)
    plt.figure().canvas.set_window_title('Training Data: Day of Week VS. Mean Count on Day')
    #plot each bar
    bar_width = 0.3
    plt.bar(day_cas.index, day_cas['mean'], bar_width, yerr=day_cas['std'], alpha=0.4, color='r', ecolor='r', label='Casual')
    plt.bar(day_reg.index + bar_width, day_reg['mean'], bar_width, yerr=day_reg['std'], alpha=0.4, color='g', ecolor='g', label='Registered')
    plt.bar(day_count.index + 2*bar_width, day_count['mean'], bar_width, yerr=day_count['std'], alpha=0.4, color='b', ecolor='b', label='Count')
    #chart labeling
    plt.xticks(day_cas.index + (1.5)*bar_width, ('Mon', 'Tue', 'Wed', 'Thur', 'Fri', 'Sat', 'Sun'))
    plt.ylabel('Mean Count')
    plt.xlabel('Day of Week')
    plt.title('Training Data: Day of Week VS. Mean Count on Day\n(Note: Hours included are from 12am to 5am)')
    plt.legend()
    plt.show()
    
    #plot by hour average count for each hour in the day on weekend and and weekdays
    hour_workday = data_train[(data_train['dayofweek'] < 5)].groupby('hour').agg({'count': [np.mean, np.std]})
    hour_weekend = data_train[(data_train['dayofweek'] > 4)].groupby('hour').agg({'count': [np.mean, np.std]})
    #droplevel statement just allows access to the mean and std columns
    hour_workday.columns = hour_workday.columns.droplevel(0)
    hour_weekend.columns = hour_weekend.columns.droplevel(0)
    plt.figure().canvas.set_window_title('Training Data: Hour in Day VS. Mean Count on Hour')
    #plot each bar
    bar_width = 0.4
    plt.bar(hour_workday.index, hour_workday['mean'], bar_width, yerr=hour_workday['std'], alpha=0.4, color='b', ecolor='b', label='Workday')
    plt.bar(hour_weekend.index + bar_width, hour_weekend['mean'], bar_width, yerr=hour_weekend['std'], alpha=0.4, color='g', ecolor='g', label='Weekend')
    #chart labeling
    plt.xticks(hour_workday.index + bar_width, hour_workday.index)
    plt.ylabel('Mean Count')
    plt.xlabel('Hour in Day')
    plt.title('Training Data: Hour in Day VS. Mean Count on Hour')
    plt.legend()
    plt.show()
    
    #plot average count for each month
    data_train_night = data_train[((data_train['hour'] > 23) | (data_train['hour'] < 6)) & (data_train['workingday'] == 1)]
    month_count = data_train_night.groupby('month').agg({'count': [np.mean, np.std]})
    #droplevel statement just allows access to the mean and std columns
    month_count.columns = month_count.columns.droplevel(0)
    plt.figure().canvas.set_window_title('Training Data: Month VS. Mean Count During Month')
    #plot each bar
    bar_width = 0.5
    plt.bar(month_count.index, month_count['mean'], bar_width, yerr=month_count['std'], align='center', alpha=0.4, color='b', ecolor='b')
    #chart labeling
    plt.xticks(month_count.index, ('Jan', 'Feb', 'Mar', 'Apr', 'May', 'June', 'July', 'Aug', 'Sept', 'Oct', 'Nov', 'Dec'))
    plt.ylabel('Mean Count')
    plt.xlabel('Month')
    plt.title('Training Data: Month VS. Mean Count During Month\n(Note: Hours included are from 12am to 5am on work days)')
    plt.show()
    
    #plot average count for each weather type
    weather_count = data_train[((data_train['hour'] > 23) | (data_train['hour'] < 6)) & (data_train['workingday'] == 1)].groupby('weather').agg({'count': [np.mean, np.std]})
    #droplevel statement just allows access to the mean and std columns
    weather_count.columns = weather_count.columns.droplevel(0)
    plt.figure().canvas.set_window_title('Training Data: Weather VS. Mean Count')
    #plot each bar
    bar_width = 0.5
    plt.bar(weather_count.index, weather_count['mean'], bar_width, yerr=weather_count['std'], align='center', alpha=0.4, color='b', ecolor='b')
    #chart labeling
    plt.xticks(weather_count.index, ('Clear/Cloudy','Mist/Cloudy','Light/Scattered Snow/Rain','Heavy Weather'))
    plt.ylabel('Mean Count')
    plt.xlabel('Weather Type')
    plt.title('Training Data: Weather VS. Mean Count\n(Note: Hours included are from 12am to 5am on work days)')
    plt.show()
    
    #plot average count for each season
    season_count = data_train[((data_train['hour'] > 23) | (data_train['hour'] < 6)) & (data_train['workingday'] == 1)].groupby('season').agg({'count': [np.mean, np.std]})
    #droplevel statement just allows access to the mean and std columns
    season_count.columns = season_count.columns.droplevel(0)
    plt.figure().canvas.set_window_title('Training Data: Season VS. Mean Count')
    #plot each bar
    bar_width = 0.5
    plt.bar(season_count.index, season_count['mean'], bar_width, yerr=season_count['std'], align='center', alpha=0.4, color='b', ecolor='b')
    #chart labeling
    plt.xticks(season_count.index, ('Spring','Summer','Fall','Winter'))
    plt.ylabel('Mean Count')
    plt.xlabel('Season')
    plt.title('Training Data: Season VS. Mean Count\n(Note: Hours included are from 12am to 5am on work days)')
    plt.show()
    
    #plot temperature vs count
    atemp_count = data_train[((data_train['hour'] > 23) | (data_train['hour'] < 6)) & (data_train['workingday'] == 1)].groupby( pd.cut(data_train['atemp'], np.arange(0, 47, 2)) ).agg({'count' : [np.mean, np.std, np.count_nonzero]})
    #droplevel statement just allows access to the mean and std columns
    atemp_count = atemp_count.head(-4)
    atemp_count.columns = atemp_count.columns.droplevel(0)
    print "\natemp_count"
    print atemp_count;
    plt.figure().canvas.set_window_title('Training Data: Apparent Temperature vs Mean Count')
    bar_width = 1.0
    bins = 0,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,32,34,36
    plt.bar(bins, atemp_count['mean'], bar_width, yerr=atemp_count['std'], align='center', alpha=0.4, color='b', ecolor='b')
    plt.title('Training Data: Apparent Temperature VS. Mean Count\n(Note: Hours included are from 12am to 5am on work days)')
    plt.ylabel('Mean Count')
    plt.xlabel('Temperature')
    plt.show()
    
    #plot actual temperature vs count
    temp_count = data_train[((data_train['hour'] > 23) | (data_train['hour'] < 6)) & (data_train['workingday'] == 1)].groupby( pd.cut(data_train['temp'], np.arange(0, 47, 2)) ).agg({'count' : [np.mean, np.std, np.count_nonzero]})
    #droplevel statement just allows access to the mean and std columns
    temp_count = temp_count.head(-7)
    temp_count.columns = temp_count.columns.droplevel(0)
    print "\ntemp_count"
    print temp_count;
    plt.figure().canvas.set_window_title('Training Data: Actual Temperature vs Mean Count')
    bar_width = 1.0
    bins = 0,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30
    plt.bar(bins, temp_count['mean'], bar_width, yerr=temp_count['std'], align='center', alpha=0.4, color='b', ecolor='b')
    plt.title('Training Data: Actual Temperature VS. Mean Count\n(Note: Hours included are from 12am to 5am on work days)')
    plt.ylabel('Mean Count')
    plt.xlabel('Temperature')
    plt.show()
    
    #plot humidity vs count
    hum_count = data_train[((data_train['hour'] > 23) | (data_train['hour'] < 6)) & (data_train['workingday'] == 1)].groupby( pd.cut(data_train['humidity'], np.arange(20, 101, 5)) ).agg({'count' : [np.mean, np.std, np.count_nonzero]})
    #droplevel statement just allows access to the mean and std columns
    #hum_count = hum_count.head(-4)
    hum_count.columns = hum_count.columns.droplevel(0)
    print "\nhum_count"
    print hum_count;
    plt.figure().canvas.set_window_title('Training Data: Humidity vs Mean Count')
    bar_width = 2.0
    bins = 20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95
    plt.bar(bins, hum_count['mean'], bar_width, yerr=hum_count['std'], align='center', alpha=0.4, color='b', ecolor='b')
    plt.title('Training Data: Humidity VS. Mean Count\n(Note: Hours included are from 12am to 5am on work days)')
    plt.ylabel('Mean Count')
    plt.xlabel('Humidity')
    plt.show()
    
    #plot windspeed vs count
    wind_count = data_train[((data_train['hour'] > 23) | (data_train['hour'] < 6)) & (data_train['workingday'] == 1)].groupby( pd.cut(data_train['windspeed'], np.arange(5, 63, 5)) ).agg({'count' : [np.mean, np.std, np.count_nonzero]})
    #droplevel statement just allows access to the mean and std columns
    wind_count = wind_count.head(-4)
    wind_count.columns = wind_count.columns.droplevel(0)
    print "\nwind_count"
    print wind_count;
    plt.figure().canvas.set_window_title('Training Data: Windspeed vs Mean Count')
    bar_width = 2.0
    bins = 5,10,15,20,25,30,35
    plt.bar(bins, wind_count['mean'], bar_width, yerr=wind_count['std'], align='center', alpha=0.4, color='b', ecolor='b')
    plt.title('Training Data: Windspeed VS. Mean Count\n(Note: Hours included are from 12am to 5am on work days)')
    plt.ylabel('Mean Count')
    plt.xlabel('Windspeed')
    plt.show()
    

if __name__=="__main__":
    main()