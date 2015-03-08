import numpy as np
import matplotlib.pyplot as plt
import csv, math
from sklearn import datasets, svm
from datetime import datetime

def main():
    
    #using reader to get all the bike sharing data from the "train.csv"
    print "Reading from train.csv file."
    reader = csv.reader(open("train.csv", "rb"), delimiter=',')
    reader_list_output = list(reader)
    training_data = np.array(reader_list_output)
    
    #parse the X_train and y_train data
    print "Parsing and filtering the training data."
    X_train = filter_X(training_data, 10887)
    y_train = filter_y(training_data, 10887)
    
    #print the training data to check values
    print X_train
    print y_train
    
    #using reader to get all the bike sharing X_test data from the "test.csv"
    print "Reading from test.csv file."
    reader = csv.reader(open("test.csv", "rb"), delimiter=',')
    reader_list_output = list(reader)
    test_data = np.array(reader_list_output)
    
    #parse the X_test data form the test_data
    print "Parsing and filtering the test data."
    X_test = filter_X(test_data, 6494)
    print X_test

#method to filter, cut, and update the X data for both training and testing
def filter_X(training_data, X_length):
    
    #find and replace datetime with exploded sin val function
    index = 0
    #iterate the X_datetime input array
    X_datetime_exploded = np.empty([X_length, 2], dtype=int)
    for x_date in X_datetime.T:
        
        #create the datetime object for this index of the array
        x_date_object = datetime.strptime(x_date.astype('str'), '%Y-%m-%d %H:%M:%S')
        x_hour = x_date_object.hour
        
        #find a day of the year, sin output tuple for each value
        day_of_year = x_date_object.timetuple().tm_yday
        X_datetime_exploded[index] = day_of_year, day_to_sin_val(day_of_year)
        index += 1;
    
    #plot the days of the year as found in train.csv
    plt.plot(X_datetime_exploded[:, 0], X_datetime_exploded[:, 1])
    plt.xlabel("Day Of The Year")
    plt.ylabel("Sinusoidal Integer Representation")
    plt.show()
    
    #remove the index and return the spliced array
    X_datetime_exploded_spliced = X_datetime_exploded[0:X_data_length, 1:2]
    return
    

#method to appropriately filter the y_train data from the training_data
def filter_y(training_data, y_length):
    return

#method to convert the number day of the year into a int value close to its sin wave seasonal value
def day_to_sin_val(day_of_year):
    #changing the period
    B = (2 * math.pi) / 365
    #derive C such that B(x - C) = pi/2 (the max of the sine wave)
    C = 81.75
    #function to return a radian value of the day of the year
    radian_output = math.sin(B * (day_of_year - C))
    #parse the float value returned into a good rounded int value
    arbitrary_multiplier = 365
    scaled_double_output = radian_output * arbitrary_multiplier
    int_output = int(float(scaled_double_output)) + 365
    return int_output

if __name__=="__main__":
    main()