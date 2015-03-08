import numpy as np
import matplotlib.pyplot as plt
import csv, math
from sklearn import datasets, svm, preprocessing
from datetime import datetime
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.svm import SVC
from sklearn import linear_model

def main():
    
    #using reader to get all the bike sharing data from the "train.csv"
    print "Reading from train.csv file."
    reader = csv.reader(open("train.csv", "rb"), delimiter=',')
    reader_list_output = list(reader)
    training_data = np.array(reader_list_output)
    
    #parse the X_train and y_train data
    print "Parsing and filtering the training data."
    min_max_scaler = preprocessing.MinMaxScaler()
    X_train = min_max_scaler.fit_transform(np.array(filter_X(training_data), dtype = 'float_'))
    y_train = filter_y(training_data)
    #np.set_printoptions(threshold=np.nan)   
     
    """
    # Split the dataset in two equal parts
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)
    """
    
    #using reader to get all the bike sharing X_test data from the "test.csv"
    print "Reading from test.csv file."
    reader = csv.reader(open("test.csv", "rb"), delimiter=',')
    reader_list_output = list(reader)
    test_data = np.array(reader_list_output)
    
    
    #parse the X_test data form the test_data
    print "Parsing and filtering the test data."    
    X_test = min_max_scaler.fit_transform(np.array(filter_X(test_data), dtype = 'float_'))
    print X_test
    
    """
    #create the linear SVM model and make a pred iction on the training set
    lin_clf = svm.LinearSVC()
    print "Using SVM and the training data to build/fit a model."
    print lin_clf.fit(X_train, y_train)
    print "Using fit to make prediction on X_test data."
    y_test = lin_clf.predict(X_test)
    """
    #create a lasso model and make a prediction on the training set
    clf = linear_model.Ridge(alpha=1.0)
    print "Using Lasso and the training data to build a model"
    clf.fit(X_train, y_train)
    print "Using fit to make prediction on X_test data"
    y_test = clf.predict(X_test)
    
    """
    #np.set_printoptions(threshold=np.nan)
    y_output = lin_clf.predict(X_test)
    print y_output
    print type(y_output)
    print y_test
    print type(y_test)
    for row_index in range(0, len(y_test)):
        print "%d: %s, %s, %s" % (row_index, str(X_test[row_index][0]), str(y_test[row_index]), str(y_output[row_index]))
    #print np.concatenate((y_output, y_test.T), axis=1) 
    """
    
    #print to see the predicted count of bikes
    print y_test
    
    #filter the predicted data and add a column for the datetime
    y_test_output = filter_predicted_data(y_test, test_data)
    print y_test_output
    
    #write results to csv
    print "Writing predictions to bike_sharing_ridge_march_8.8.csv"
    results = open('bike_sharing_ridge_march_8.8.csv', 'w')
    #header required by Kaggle: datetime,count
    results.write('datetime,count')
    #rows (predictions)
    for index in range (0,6493):
        results.write('\n')
        results.write('%s,' % y_test_output[index][0])
        results.write(str(y_test_output[index][1]))
    print "Done."

#method to filter, cut, and update the X data for both training and testing
def filter_X(training_data):
    
    #output will be of form: {day sin val, hour sin val, season, holiday, workday, weather, temp, atemp, humidity, windspeed }
    X_train = np.empty([len(training_data) - 1, 10], dtype=float)
    
    #find the output by indexing and filtering/exploding
    for row_index in range(0, len(training_data) - 1):
        
        #find the sin values for exploded datetime day and hour, row_index+1 to account for the title row in train.csv
        date_object = datetime.strptime(training_data[row_index+1][0].astype('str'), '%Y-%m-%d %H:%M:%S')        
        X_train[row_index][0] = day_to_sin_val(date_object.timetuple().tm_yday)
        X_train[row_index][1] = 100*hour_to_sin_val(date_object.hour)
        
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
def filter_predicted_data(y_test, test_data):
    y_output = np.empty([(len(test_data) - 1), 2], dtype=object)
    print len(y_output)
    print len(test_data)
    print len(y_test)
    for row_index in range(0, len(test_data) - 1):
        #reverse the natural log to get an int value
        y_output[row_index] = test_data[row_index+1][0], math.exp(float(y_test[row_index]))
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

if __name__=="__main__":
    main()