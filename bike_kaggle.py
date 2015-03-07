import numpy as np
import matplotlib.pyplot as plt
import csv, math
from sklearn import datasets, svm
from datetime import datetime

#method to convert the number day of the year into a int value close to its sin wave seasonal value
def day_to_sin_val(day_of_year):
    #changing the period
    B = (2 * math.pi) / 365
    #derive C such that B(x - C) = pi/2 (the max of the sine wave)
    C = 91.75
    #function to return a radian value of the day of the year
    radian_output = math.sin(B * (day_of_year - C))
    #parse the float value returned into a good rounded int value
    arbitrary_multiplier = 365
    scaled_double_output = radian_output * arbitrary_multiplier
    int_output = int(float(scaled_double_output)) + 365
    return int_output
 
#find and replace datetime with sanitized parameters
def explode_datetime_and_graph(X_datetime, X_data_length):
    index = 0
    #iterate the X_datetime input array
    X_datetime_exploded = np.empty([X_data_length, 26], dtype=int)
    for x_date in X_datetime.T:
        
        #create the datetime object for this index of the array
        x_date_object = datetime.strptime(x_date.astype('str'), '%Y-%m-%d %H:%M:%S')
        
        #add a boolean vector for each hour
        values = 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0
        h0,h1,h2,h3,h4,h5,h6,h7,h8,h9,h10,h11,h12,h13,h14,h15,h16,h17,h18,h19,h20,h21,h22,h23 = values
        x_hour = x_date_object.hour
        if x_hour == 0:
            h0 = 1
        if x_hour == 1:
            h1 = 1
        if x_hour == 2:
            h2 = 1
        if x_hour == 3:
            h3 = 1
        if x_hour == 4:
            h4 = 1
        if x_hour == 5:
            h5 = 1
        if x_hour == 6:
            h6 = 1
        if x_hour == 7:
            h7 = 1
        if x_hour == 8:
            h8 = 1
        if x_hour == 9:
            h9 = 1
        if x_hour == 10:
            h10 = 1
        if x_hour == 11:
            h11 = 1
        if x_hour == 12:
            h12 = 1
        if x_hour == 13:
            h13 = 1
        if x_hour == 14:
            h14 = 1
        if x_hour == 15:
            h15 = 1
        if x_hour == 16:
            h16 = 1
        if x_hour == 17:
            h17 = 1
        if x_hour == 18:
            h18 = 1
        if x_hour == 19:
            h19 = 1
        if x_hour == 20:
            h20 = 1
        if x_hour == 21:
            h21 = 1
        if x_hour == 22:
            h22 = 1
        if x_hour == 23:
            h23 =1
        
        #find a day of the year, sin output tuple for each value
        day_of_year = x_date_object.timetuple().tm_yday
        X_datetime_exploded[index] = day_of_year, day_to_sin_val(day_of_year),h0,h1,h2,h3,h4,h5,h6,h7,h8,h9,h10,h11,h12,h13,h14,h15,h16,h17,h18,h19,h20,h21,h22,h23
        index += 1;
    
    #plot the days of the year as found in train.csv
    plt.plot(X_datetime_exploded[:, 0], X_datetime_exploded[:, 1])
    plt.xlabel("Day Of The Year")
    plt.ylabel("Sinusoidal Integer Representation")
    plt.show()
    
    #remove the index and return the spliced array
    X_datetime_exploded_spliced = X_datetime_exploded[0:X_data_length, 1:26]
    return X_datetime_exploded_spliced
    
#method to filter, cut, and update the X data for both training and testing
def filter_X(data, X_length):
    #find and replace datetime with exploded parameters: datetime => {sin_val, hour1_vector, hour2_vector,.... hour23_vector} 
    X_datetime = data[1:X_length, 0]
    X_datetime_exploded = explode_datetime_and_graph(X_datetime, (X_length - 1))
    
    #take the original data in integer form and concatenate the exploded parameters
    #removed seasons as sin wave for day of year has been added, removed casual and registered ******
    X_train = data[1:X_length, 2:11]
    
    #type cast the X_train values from strings to needed types
    X_train_filtered = np.empty([(X_length - 1), 32], dtype=object)
    index = 0
    for X in X_train:
        X_train_filtered[index][0] = X_train[index][0].astype('int')
        X_train_filtered[index][1] = X_train[index][1].astype('int')
        X_train_filtered[index][2] = X_train[index][2].astype('int')
        X_train_filtered[index][3] = X_train[index][3].astype('float')
        X_train_filtered[index][4] = X_train[index][4].astype('float')
        X_train_filtered[index][5] = X_train[index][5].astype('int')
        X_train_filtered[index][6] = X_train[index][6].astype('float')
        X_train_filtered[index][7] = X_datetime_exploded[index][0]
        X_train_filtered[index][8] = X_datetime_exploded[index][1]
        X_train_filtered[index][9] = X_datetime_exploded[index][2]
        X_train_filtered[index][10] = X_datetime_exploded[index][3]
        X_train_filtered[index][11] = X_datetime_exploded[index][4]
        X_train_filtered[index][12] = X_datetime_exploded[index][5]
        X_train_filtered[index][13] = X_datetime_exploded[index][6]
        X_train_filtered[index][14] = X_datetime_exploded[index][7]
        X_train_filtered[index][15] = X_datetime_exploded[index][8]
        X_train_filtered[index][16] = X_datetime_exploded[index][9]
        X_train_filtered[index][17] = X_datetime_exploded[index][10]
        X_train_filtered[index][18] = X_datetime_exploded[index][11]
        X_train_filtered[index][19] = X_datetime_exploded[index][12]
        X_train_filtered[index][20] = X_datetime_exploded[index][13]
        X_train_filtered[index][21] = X_datetime_exploded[index][14]
        X_train_filtered[index][22] = X_datetime_exploded[index][15]
        X_train_filtered[index][23] = X_datetime_exploded[index][16]
        X_train_filtered[index][24] = X_datetime_exploded[index][17]
        X_train_filtered[index][25] = X_datetime_exploded[index][18]
        X_train_filtered[index][26] = X_datetime_exploded[index][19]
        X_train_filtered[index][27] = X_datetime_exploded[index][20]
        X_train_filtered[index][28] = X_datetime_exploded[index][21]
        X_train_filtered[index][29] = X_datetime_exploded[index][22]
        X_train_filtered[index][30] = X_datetime_exploded[index][23]
        X_train_filtered[index][31] = X_datetime_exploded[index][24]
        index += 1
        
    #return the filterd X data
    return X_train_filtered

#method to appropriately filter the y_train data from the training_data
def filter_y(training_data, y_length):
    
    #parse the y training data
    y_train = training_data[1:y_length, 11]
    
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
def filter_predicted_data(y_test, test_data, y_length):
    y_output = np.empty([(y_length - 1), 2], dtype=object)
    y_train_datetime = test_data[1:y_length, 0]
    index = 0
    for y in y_test:
        #reverse the natural log to get an int value
        y_output[index] = y_train_datetime[index], math.exp(float(y_test[index]))
        index += 1
    #reverse the natural log!
    return y_output

#method to write the predicted y data to a csv "bike_sharing_output.csv"
def write_prediction_to_csv():
    return 0

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
    
    
    #create the linear SVM model and make a pred iction on the training set
    lin_clf = svm.LinearSVC()
    print "Using SVM and the training to build/fit a model."
    lin_clf.fit(X_train, y_train)
    print "Using fit to make prediction on X_test data."
    y_test = lin_clf.predict(X_test)
    
    #print to see the predicted count of bikes
    print y_test
    
    #filter the predicted data and add a column for the datetime
    y_test_output = filter_predicted_data(y_test, test_data, 6494)
    print y_test_output
    
    #write results to csv
    print "Writing predictions to bike_sharing_svm_linear.csv"
    results = open('bike_sharing_svm_linear.csv', 'w')
    #header required by Kaggle: datetime,count
    results.write('datetime,count')
    #rows (predictions)
    for index in range (0,6493):
        results.write('\n')
        results.write('%s,' % y_test_output[index][0])
        results.write(str(y_test_output[index][1]))
    print "Done."
    

if __name__=="__main__":
    main()
