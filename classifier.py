#!/usr/bin/env python3

'''
Jennefer Maldonado
Final Project for CSC 590
Detect Hepatitis C disease using blood biomarkers
'''

import time
import math
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt

'''
To Do:
    tables and plots
    implementation of appropriate classifier
'''

'''
Missing Value Treatment
'''
def clean_data(data):
    num_rows = len(data)
    rows_to_drop = []
    for i in range(0, num_rows):
        current_row = data.iloc[i]
        for value in current_row:
            if value == 'NA':
                rows_to_drop.append(i)
    # make list unique
    rows_to_drop = list(set(rows_to_drop))
    clean_dataset = data.drop(rows_to_drop)
    return clean_dataset


'''
Summary Statistics:
    used to understand the data well and get an overall
    picture of what we expect the data to look like
'''

def mean(data):
    '''
    Modified mean method from Naive Bayes Classifer
    '''
    mean_dict = {}
    for (col_name, col_data) in data.iteritems():
        if all ([col_name != 'Category', col_name != 'Unnamed: 0', col_name != 'Sex']):
            col_data = col_data.to_numpy()
            col_data = col_data.astype(np.float64)
            col_mean = np.mean(col_data)
            mean_dict[col_name] = col_mean
    return mean_dict

def standard_dev(data):
    std_dict = {}
    for (col_name, col_data) in data.iteritems():
        if all ([col_name != 'Category', col_name != 'Unnamed: 0', col_name != 'Sex']):
            col_data = col_data.to_numpy()
            col_data = col_data.astype(np.float64)
            std_dev = np.std(col_data)
            std_dict[col_name] = std_dev
    return std_dict

def normalize(data):
    '''
    Using Minimum and Maximum of column data
    '''
    norm_dataframe = data.copy()
    for (col_name, col_data) in data.iteritems():
        if all ([col_name != 'Category', col_name != 'Unnamed: 0', col_name != 'Sex']):
            current_col = pd.to_numeric(data[col_name])
            maximum = current_col.max()
            minimum = current_col.min()
            norm_dataframe[col_name] = (current_col- minimum)/(maximum-minimum)
    return norm_dataframe

def covariance(col_x,col_y):
    col_x = col_x.to_numpy()
    col_y = col_y.to_numpy()
    n = len(col_x)
    multiplier = 1/(n-1)
    x_bar = np.mean(col_x)
    y_bar = np.mean(col_y)
    total_sum = 0
    for i in range(0,n):
        total_sum += (col_x[i]- x_bar)*(col_y[i]-y_bar)
    return (multiplier*total_sum)

def correlation(col_x,col_y):
    cov =  covariance(col_x,col_y)
    std_dev_x = np.std(col_x)
    std_dev_y = np.std(col_y)
    return (cov/(std_dev_x*std_dev_y))


'''
print(data_frame.head())
print(data_frame.iloc[0])
print()
print(df_train.iloc[0])
print()
print(df_train.head())
print(df_test.iloc[0])
print()
print(df_test.head())
'''

data_frame = pd.read_csv('hcvdat0.csv', keep_default_na=False)

# remove rows with missing values
data_frame = clean_data(data_frame)

# converts all males to category 1 and females to category 0
data_frame['Sex'] = data_frame['Sex'].astype('category').cat.codes

'''
Mean and Standard Deviation are very far for each column 
    Approach: Normalize all data in the dataset columns 
    that are not category, name, and sex
'''
means = mean(data_frame)
stddevs = standard_dev(data_frame)
print('Unnormed means and standard deviation')
print(means)
print(stddevs)
norm_dataframe = normalize(data_frame)
norm_means = mean(norm_dataframe)
norm_std = standard_dev(norm_dataframe)
print('Normalized means and standard deviation')
print(norm_means)
print(norm_std)

df_train = norm_dataframe.sample(frac=1/3)
df_test = norm_dataframe[~(norm_dataframe.index.isin(df_train.index))]

'''
Now all data is inbetween [0,1] 
Compute correlation between all numeric columns
'''
column_names = ['Age', 'ALB', 'ALP', 'ALT', 'AST', 'BIL', 'CHE', 'CHOL', 'CREA', 'GGT', 'PROT']
for i in range(0, len(column_names)):
    col_i = df_train[column_names[i]]
    for j in range(0, i):
        col_j = df_train[column_names[j]]
        current_cor = correlation(col_i,col_j)
        if current_cor< -0.5 or current_cor >0.5:
            print(''+str(column_names[i])+', '+ str(column_names[i])+ ': ' +str(current_cor))
'''
All correlation coefficients are around zero which means we are safe to use 
KNN and Naive Bayes. No set of columns is correlated with each other.
Make a few scatter plots
for i in range(0, len(column_names)):
    col_i = df_train[column_names[i]]
    for j in range(0, i):
        col_j = df_train[column_names[j]]
        plt.scatter(col_i,col_j) 
        plt.yticks(np.arange(col_j.min(), col_j.max()+0.05, 0.05))
        plt.title('X:' + str(column_names[i]) + ' Y:' + str(column_names[j]))
        plt.show()
'''

'''
KNN Implementation
'''
 # want to ignore col 0, 1, and 3

def euclid_dist(v1, v2):
    '''
    Computes the Euclidean Distance of two vectors v1 and v2
    Ignores the name and type column in distance calculation
    Params:
    v1 - list of attribute values
    v2 - list of attribute values
    Returns the euclidean distance of the two vectors
    '''
    dist = 0
    for vi in range(2, len(v1)):
        dist += (v1[vi] - v2[vi])**2
    return math.sqrt(dist)

def find_neighbors(current_vector, possible_neighbors, max_neighbors):
    '''
    Finds all possible neighbors by storing the distances in a dictionary,
    sorting the dictionary and finding the first k neighbors with the closest distances
    Params: 
    current_vector - the row we are checking 
    possible_neighbors - the set of possible rows to check against
    max_neighbors - the maximum number of neighbors per cluster
    '''
    all_distances = {}
    for row in possible_neighbors:
        all_distances[euclid_dist(current_vector, row)] = row
    sorted_dists = dict(sorted(all_distances.items(), key=lambda item: item[0]))
    keys = list(sorted_dists.values())
    return keys[:max_neighbors]

def predict(current_vector, possible_neighbors, max_neighbors):
    '''
    Calls find_neighbors and then slices the class labels out
    Finds the maximum of the class counts and returns the label for that class
    Params: 
    current_vector - the row we are checking 
    possible_neighbors - the set of possible rows to check against
    max_neighbors - the maximum number of neighbors per cluster
    '''
    current_neighbors = find_neighbors(current_vector, possible_neighbors, max_neighbors)
    classes = [row[1:2] for row in current_neighbors]
    return max(classes, key=classes.count)

def accuracy(actual, predicted):
    '''
    Computes the accuracy of the given model
    Params: 
    actual - list of actual values
    predicted - list of predicted values
    Returns the percent of matching values
    '''
    total = 0
    for i in range(len(predicted)):
        if predicted[i] == actual[i]:
            total+=1
    return (float(total)/float(len(predicted))) * 100.0


knn_df_train = df_train[:].values
knn_df_test = df_test[:].values

# loop through the number of neighbors and the rows
k_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
for k in k_list:
    actual_labels = []
    predicted_labels = []
    start_time = time.time()
    for row in knn_df_train:
        actual_labels.append(row[1])
        output = predict(row, knn_df_test, k)
        predicted_labels.append(output)
    print('The accuracy for k='+ str(k)+ ' is: ' + str(accuracy(actual_labels,predicted_labels)))
    print("Ran for %s seconds" % (time.time() - start_time))


'''
Naive Bayes Implementation
'''

def statistics(training_data):
    '''
    Computes the length, mean, and standard deviation of each class in the training data
    Params:
    training_data - the data we wish to analyze
    Returns a dictionary with the class name and it's information in a list
    '''
    class_stats = {}
    df_class_type = training_data.groupby('Category')
    # iris dataset only has 3 classes
    classes = ['0=Blood Donor','0s=suspect Blood Donor','1=Hepatitis','2=Fibrosis','3=Cirrhosis']
    for c in classes:
        # sort dataframe by class type
        current_class_data = df_class_type.get_group(c)
        # drop type column for statistics
        current_class_data = current_class_data.drop(['Category','Unnamed: 0','Sex'],1)
        # store the length of the class data for later use
        col_stats = [len(current_class_data)]
        # loop through each attribute and find the mean and standard deviation
        for (col_name, col_data) in current_class_data.iteritems():
            col_data = col_data.to_numpy()
            col_data = col_data.astype(float)
            mean = np.mean(col_data)
            std_dev = np.std(col_data)
            col_stats.append([mean,std_dev])
        class_stats[str(c)] = col_stats
    return class_stats

def pdf(x, mean, std_dev):
    '''
    Computes the relative likelihood that another value
    would be close to the given value
    This helps in finding what class a given sample would fall into
    We are using normal distribution for this classifier
    Params:
    x -  the value we want to check the relative likelihood of
    mean - mean of the class we want to check against
    std_dev - standard deviation of the class we want to check against
    Returns the likelihood value 
    '''
    y = math.exp(-((x-mean)**2 / (2*std_dev**2)))
    return (1/ (math.sqrt(2*math.pi)*std_dev )) * y

def compute_probability(current_row, class_statistics):
    '''
    Computes the class probabilities for the current row 
    Returns a dictionary of the probabilities for each class labeled with the class label
    Params:
    current_row - the row to check 'closeness' of 
    class_statistics - the dictionary holding the number of instances per class and the
        mean and standard deviation of each column of the class training dataset
    '''
    class_probabilties = {}
    # gets the class label and then values holds the list of [length, [mean, std], [mean,std]...]
    for class_label, values in class_statistics.items():
        # this computes the P(class_label) and stores it as the initial value in the dictionary
        class_probabilties[class_label] = float(values[0])/total_rows
        # now find the pdf of each value in the current row using the mean and standard deviation of each column
        for i in range(2, len(values)):
            # 1 skips the length stored in the first position
            if i != 3:
                mean, std_dev = values[i]
                class_probabilties[class_label] = class_probabilties[class_label] * pdf(float(current_row[i]), mean, std_dev)
    return class_probabilties

def naive_prediction(current_row, class_statistics):
    '''
    Finds the probabilites per class and find the maximum one, this becomes the class label
    Params:
    current_row - the row to check 'closeness' of 
    class_statistics - the dictionary holding the number of instances per class and the
        mean and standard deviation of each column of the class training dataset
    Returns the predicted class label
    '''
    probabilities = compute_probability(current_row, class_statistics)
    class_label = max(probabilities, key=probabilities.get)
    return class_label



nb_train = data_frame.sample(frac=1/3)
nb_test = data_frame[~(data_frame.index.isin(nb_train.index))]


'''
total_rows = len(nb_train)
class_stats = statistics(nb_train)
nb_test = nb_test[:].values
# empty lists for accuracy checking
naive_predictions = []
naive_actual = []
# loop through each data instance
for row in nb_test:
    c = naive_prediction(row, class_stats)
    naive_predictions.append(c)
    naive_actual.append(row[1])
# output results
print('Naive Bayes classifier has accuracy: ' + str(accuracy(naive_actual, naive_predictions)))
'''


total_rows = len(df_train)
class_stats = statistics(df_train)
df_test = df_test[:].values
# empty lists for accuracy checking
naive_predictions = []
naive_actual = []
# loop through each data instance
for row in df_test:
    c = naive_prediction(row, class_stats)
    naive_predictions.append(c)
    naive_actual.append(row[1])
# output results
print('Naive Bayes classifier has accuracy: ' + str(accuracy(naive_actual, naive_predictions)))
