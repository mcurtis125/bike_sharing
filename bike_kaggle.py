import numpy as np
import matplotlib.pyplot as plt
import csv, math
from sklearn import datasets, svm, preprocessing, linear_model, ensemble
from datetime import datetime
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.svm import SVC
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import ElasticNet, ElasticNetCV, LassoLarsIC, LassoLarsCV, LarsCV

#method to change the predictor we are using
def predict(X_train, y_train, X_test):
    
    #create a model and make a prediction on the training set
    clf = ensemble.GradientBoostingRegressor(n_estimators=1000, learning_rate=0.01, max_depth=3, min_samples_split=1, loss='ls')
    #clf = ensemble.GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=1, random_state=0, loss='ls')
    print "Using Algorithm and the training data to build a model"
    clf.fit(X_train, y_train)
    #print "clf.alpha: %f" %(clf.alpha)
    print "Using fit to make prediction on X_test data"
    y_test = clf.predict(X_test)
    
    #filter the predicted data to reverse the natural log
    return filter_predicted_data(y_test)

def main():
    
    #using reader to get all the bike sharing data from the "train.csv"
    print "Reading from train.csv file."
    reader = csv.reader(open("train.csv", "rb"), delimiter=',')
    reader_list_output = list(reader)
    data = np.array(reader_list_output)
    
    #in this case we want to define our test data as a split of the training data
    test_data = data[-1000:]
    training_data = data[:(len(data) - len(test_data))]
    
    #filtering the training and test data
    print "Parsing and filtering the test and training data."
    X_train_filtered = np.array( filter_X_Simple(training_data, header_present=True), dtype='float' )
    X_test_filtered = np.array( filter_X_Simple(test_data), dtype='float' )
    y_train = np.array( filter_y(training_data), dtype='float' )
        
    print "Normalizing the filtered test and training data"
    #note that this fit_trasnform function is normalizing the data to the range [0,1]
    #min_max_scaler = preprocessing.MinMaxScaler()
    #X_train = min_max_scaler.fit_transform(X_train_filtered)
    #X_test = min_max_scaler.fit_transform(X_test_filtered)
    X_train = tune_X(X_train_filtered)
    X_test = tune_X(X_test_filtered)
    
    #make a prediction with the defined algorithm
    predicted = predict(X_train, y_train, X_test)
    
    #parse the actual resulting data
    actual = parse_actual(test_data)
    
    #find the difference between the predicted and actual count for each index
    difference = parse_difference(predicted, actual)
    
    #concatonate the desired information for csv review    
    to_csv = np.hstack((X_test, predicted, actual, difference))
    
    #write results to csv
    print "Writing predictions to bike_sharing_output.csv"
    results = open('bike_sharing_output.csv', 'w')
    #header are as follows:
    results.write('month,day_of_week,hour,predicted,actual,log_sq_diff\n')
    np.savetxt(results, to_csv, delimiter=',', fmt='%s')

    #evaluate the predictive algorithm method
    evaluation(predicted, actual)

    """
    #checking normalization and filtering of day and hour sin vals
    for row in range(0, len(X_test), 100):
        print "datetime: %s" % ( str(test_data[row][0]) )
        print "day_in_year: %s\thour_in_day: %s" % ( str(X_test_filtered[row][0]), str(X_test_filtered[row][1]) )
        print "norm_day_in: %s\tnorm_hor_in: %s\t" % ( str(X_test[row][0]), str(X_test[row][1]) )
    """

#method to tune the input data
def tune_X(X):
    X_tuned = X
    for row_index in range(0, len(X)):
        X_tuned[row_index][0] = 1*X[row_index][0]
        X_tuned[row_index][1] = 1*X[row_index][1]
    return X_tuned

def filter_X_Simple(training_data, header_present=False):
    #depending on if we want to get rid of the header we have set the flag header present to true, skip first row
    
    if header_present:
    
        #output will be of form: {day sin val, hour sin val, season, holiday, workday, weather, temp, atemp, humidity, windspeed }
        X_train = np.empty([len(training_data) - 1, 4], dtype=float)
    
        #find the output by indexing and filtering/exploding
        for row_index in range(0, len(training_data) - 1):
        
            #find the sin values for exploded datetime day and hour, row_index+1 to account for the title row in train.csv
            date_object = datetime.strptime(training_data[row_index+1][0].astype('str'), '%Y-%m-%d %H:%M:%S')        
        
            X_train[row_index][0] = date_object.month
            X_train[row_index][1] = date_object.isoweekday()
            X_train[row_index][2] = date_object.hour
            X_train[row_index][3] = training_data[row_index+1][4]
            
        #return the filtered/exploded data
        return X_train
    
    #if the method reaches this point then we know a header wasn't present and we can parse appropriately
    X_train = np.empty([len(training_data), 4], dtype=float)
    
    #find the output by indexing and filtering/exploding
    for row_index in range(0, len(training_data) - 1):
    
        #find the sin values for exploded datetime day and hour, row_index+1 to account for the title row in train.csv
        date_object = datetime.strptime(training_data[row_index][0].astype('str'), '%Y-%m-%d %H:%M:%S')        
        
        X_train[row_index][0] = date_object.month
        X_train[row_index][1] = date_object.isoweekday()
        X_train[row_index][2] = date_object.hour
        X_train[row_index][3] = training_data[row_index][4]

    #return the filtered/exploded data
    return X_train

#method to filter, cut, and update the X data for both training and testing
def filter_X(training_data, header_present=False):
     
    #depending on if we want to get rid of the header we have set the flag header present to true, skip first row
    if header_present:
    
        #output will be of form: {day sin val, hour sin val, season, holiday, workday, weather, temp, atemp, humidity, windspeed }
        X_train = np.empty([len(training_data) - 1, 10], dtype=float)
    
        #find the output by indexing and filtering/exploding
        for row_index in range(0, len(training_data) - 1):
        
            #find the sin values for exploded datetime day and hour, row_index+1 to account for the title row in train.csv
            date_object = datetime.strptime(training_data[row_index+1][0].astype('str'), '%Y-%m-%d %H:%M:%S')        
            X_train[row_index][0] = day_to_sin_val(date_object.timetuple().tm_yday)
            #print "datetime: %s\t tm_yday: %s\t day_to_sin_val: %s" %(str(date_object),str(date_object.timetuple().tm_yday),X_train[row_index][0])
            X_train[row_index][1] = hour_to_sin_val(date_object.hour)
        
            #X_train[row_index][0] = date_object.month
            #X_train[row_index][1] = date_object.hour
        
            #just translate all other information untouched into the filtered data
            X_train[row_index][2] = training_data[row_index+1][1]
            X_train[row_index][3] = training_data[row_index+1][2]
            X_train[row_index][4] = training_data[row_index+1][3]
            X_train[row_index][5] = training_data[row_index+1][4]
            X_train[row_index][6] = training_data[row_index+1][5]
            X_train[row_index][7] = training_data[row_index+1][6]
            X_train[row_index][8] = training_data[row_index+1][7]
            X_train[row_index][9] = training_data[row_index+1][8]
    
        #return the filtered/exploded data
        return X_train
    
    #if the method reaches this point then we know a header wasn't present and we can parse appropriately
    X_train = np.empty([len(training_data), 10], dtype=float)
    
    #find the output by indexing and filtering/exploding
    for row_index in range(0, len(training_data) - 1):
    
        #find the sin values for exploded datetime day and hour, row_index+1 to account for the title row in train.csv
        date_object = datetime.strptime(training_data[row_index][0].astype('str'), '%Y-%m-%d %H:%M:%S')        
        X_train[row_index][0] = day_to_sin_val(date_object.timetuple().tm_yday)
        #print "datetime: %s\t tm_yday: %s\t day_to_sin_val: %s" %(str(date_object),str(date_object.timetuple().tm_yday),X_train[row_index][0])
        X_train[row_index][1] = hour_to_sin_val(date_object.hour)
    
        #X_train[row_index][0] = date_object.month
        #X_train[row_index][1] = date_object.hour
    
        #just translate all other information untouched into the filtered data
        X_train[row_index][2] = training_data[row_index][1]
        X_train[row_index][3] = training_data[row_index][2]
        X_train[row_index][4] = training_data[row_index][3]
        X_train[row_index][5] = training_data[row_index][4]
        X_train[row_index][6] = training_data[row_index][5]
        X_train[row_index][7] = training_data[row_index][6]
        X_train[row_index][8] = training_data[row_index][7]
        X_train[row_index][9] = training_data[row_index][8]

    #return the filtered/exploded data
    return X_train

#method to appropriately filter the y_train data from the training_data
def filter_y(training_data):
    
    #parse the y training data
    y_train = training_data[1:len(training_data), 11]
    
    #update the y data as log(y_value) to account for the competition marking criteria: natural log-mean-squared
    #y_train_filtered = np.empty([(y_length - 1), 1], dtype=float)
    y_train_filtered = y_train
    index = 0
    for y in y_train:
        y_train_filtered[index] = math.log(float(y))
        index += 1
    
    #return the filtered y data
    return y_train_filtered
   
#method to filter the predicted y data for output in a csv, reversing the natural log of the outputs
def filter_predicted_data(y_test):
    y_output = np.empty([len(y_test), 1], dtype=object)
    for row_index in range(0, len(y_test) - 1):
        #reverse the natural log to get an int value
        y_output[row_index] = math.exp(float(y_test[row_index]))
        #print "y_output[%d]: %f " %(row_index, y_output[row_index])
    #reversed the natural log!
    return y_output
    
#method to convert the number day of the year into a int value close to its sin wave seasonal value
def day_to_sin_val(day_of_year):
    #changing the period
    B = (2 * math.pi) / 365
    #derive C such that B(x - C) = pi/2 (the max of the sine wave)
    C = 91.75
    #function to return a radian value of the day of the year
    return math.sin(B * (day_of_year - C))

#method to convert the hour in a day into a sin value
def hour_to_sin_val(hour):
    B = (2 * math.pi) / 24
    #derive C such that B(x - C) = pi/2 (the max of the sine wave), x - C = pi/(2B), C = x - pi/(2B)
    C = 6
    return math.sin(B* (hour - C))

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
    
#parse the test data to return a np array of the actual counts at each datetime
#NOTE: this method assumes that the header row has been deleted or is not present
def parse_actual(test_data):
    actual = np.empty([len(test_data), 1], dtype=float)
    for row_index in range(0, len(actual)):
        actual[row_index][0] = float(test_data[row_index][11])
    return actual

#use the predicted and actual count to return an np array showing the log_sq_diff between the values at each index
def parse_difference(predicted, actual):    
    difference = np.empty([len(predicted), 1], dtype=float)
    for row_index in range(0, len(difference) - 1):
        #log squared difference will be help in column two of the actual results
        pi = float(predicted[row_index][0])
        ai = float(actual[row_index][0])
        #print "row_index: %d pi: %f, ai: %f" %(row_index, pi, ai)
        difference[row_index][0] = log_sq_diff(pi, ai)
    return difference

if __name__=="__main__":
    main()