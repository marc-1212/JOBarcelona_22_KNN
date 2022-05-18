import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split



# functions:

# points the result of knn
def knn_f1score(n, insect_test_x, insect_test_y, insect_train_x, insect_train_y):
    #trains the algorithm
    classifier = KNeighborsClassifier(n_neighbors=n)
    classifier.fit(insect_train_x, insect_train_y)

    #predict the test dataset
    predicted = classifier.predict(insect_test_x)
    
    return (f1_score(insect_test_y, predicted, average='macro') * 100)

#change the hours to minutes and it plus to the minutes column just having left 1 column for the time
def cleaning_data(insect_dataset):
    insect_dataset['Minutes'] = insect_dataset['Minutes'] + (insect_dataset['Hour'] * 60)
    del insect_dataset['Hour']
    return insect_dataset

#separes the insect data as y and the remain datasets as x
def separating_x_and_y(insect_dataset):
    x = insect_dataset.iloc[:,:7]
    y = insect_dataset.loc[:,'Insect']
    return x, y

#finding the best K for KNN algorithm
def finding_best_k(insect_test_x, insect_test_y, insect_train_x, insect_train_y):
    k_list:list = []
    score:list = []
    for i in range(1,20):
        k_list.append(i)
        score.append(knn_f1score(i, insect_test_x, insect_test_y, insect_train_x, insect_train_y))
    total_score = pd.DataFrame({'k_list' : k_list,  'SCORE':score})
    sorted_values_by_score =  total_score.sort_values(by='SCORE', ascending=False)
    return int(sorted_values_by_score['k_list'].loc[sorted_values_by_score.index[0]])

#predicts the test dataset
def predict_with_knn(k, x_train, y_train , insect_dataset_test):
    classifier = KNeighborsClassifier(n_neighbors=k)
    classifier.fit(x_train, y_train)
    y_predicted = classifier.predict(insect_dataset_test)
    return y_predicted




# main:

#imort the data
insect_dataset_train = pd.read_csv("train.csv", index_col=0, parse_dates=True)
insect_dataset_test = pd.read_csv("test_x.csv", index_col=0, parse_dates=True)

#cleaning data
insect_dataset_train = cleaning_data(insect_dataset_train)
insect_dataset_test = cleaning_data(insect_dataset_test)

#separates x and target from the test dataset for the train
x, y = separating_x_and_y(insect_dataset_train)

#randomize the insect dataset test and splits it due to havea training dataset for the KNN algorithm and another one for testing the KNN
insect_train_x, insect_test_x, insect_train_y, insect_test_y = train_test_split(x, y, test_size=0.2, random_state=2020)

#find best k
k = finding_best_k(insect_test_x, insect_test_y, insect_train_x, insect_train_y)

#predicts dataset
prediction = predict_with_knn(k, x, y,  insect_dataset_test)

#create the dataset from result
result = pd.DataFrame({'Test_index': insect_dataset_test.index, 'Prediction': prediction})

#export the csv
result.to_csv('results.csv', index = False)