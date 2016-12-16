"""
Predicting the results of our SVM Model
"""

import numpy as np
from Get_Data import get
# get is some method which returns samples and their labels
import os
from sklearn import preprocessing, metrics
import csv
from sklearn.feature_selection import chi2, SelectKBest
try:
    import cPickle as pickle
except:
    import pickle


def netsav(recall, filtered):
    """
    Outputs the monetary values for the model
    :param recall:
    :param filtered:
    :return: Savings from infrastructure reduction, Net RTB Income Lost, and Net Savings
    """
    InfrastruceReduction = 123000*filtered
    IncomeLost = 600000*(1-recall)
    NetSavings = InfrastruceReduction - IncomeLost - 5200
    return InfrastruceReduction, IncomeLost, NetSavings


def data_format(test_day):
    """
    Changes the labels from {0,1} to {-1,1}, performs feature selection, and scales the testing samples
    :param test_day: The address of the day we wish to test on
    :return: The test set and its labels
    """

    X, y = get(addr_day=test_day)
    n = np.size(y, 0)
    y_test = 2*y[:,-1]-np.ones(n)
    # We change any zeros in the label to negative ones
    X_test = BestK.transform(X)
    # We select the K Best features only
    scaler.transform(X_test)
    # We scale each feature in a standardized way
    return X_test, y_test


def predictor(test_day, train_day):
    """
    Predicting the outcome of a bid request
    :param test_day: The address of the day we wish to test on
    :param train_day: The address of the day we wish to get the model from
    :return: The results to be written to the output file
    """
    X_test, y_test = data_format(test_day=test_day)
    clf = pickle.load(open( os.path.join(train_day,"Models/SVM.p"), "rb"))
    # We load in our fitted model
    y_pred = clf.predict(X_test)
    # The predicted results
    recall = metrics.recall_score(y_test, y_pred)
    n = len(y_test)
    filtered = sum(np.count_nonzero(y_pred-np.ones(n))) / float(n)
    # The filter rate is the proportion of the data we predict to be -1
    result = [train_day, test_day, recall, filtered, netsav(recall,filtered)]
    # We write this as a list so that it will be on one row of the csv
    return result



if __name__ == "__main__":
    with open("/file_location/SVM-Results", "w") as output_file:
        wr = csv.writer(output_file, quoting = csv.QUOTE_MINIMAL)
        # Write the results to a csv
        wr.writerow(['Train Day', 'Test Day','Recall','Filter Rate',
                     'Savings from infrastructure reduction','Net RTB Income Lost','Net Savings'])
        # Heading the rows
        for month in range(5,7):

            root = '/mnt/rips2/2016'
            p0 = "0" + str(month)
            Data , label = get('chosen day')
            # Here we standardise feature selection and scaling
            BestK = SelectKBest(chi2, k = 350)
            BestK.fit(Data, label)
            # Feature Selection - Select K Best
            Data = BestK.transform(Data)
            scaler = preprocessing.StandardScaler().fit(Data)
            # The method via which we scale the data
            for day in range(1,32):
                try:
                    p1 = str(day).rjust(2,'0')
                    p2 = str(day-1).rjust(2,'0')
                    test_day = os.path.join(root,p0,p1)
                    train_day = os.path.join(root,p0,p2)
                    # By default we perform daily training/testing
                    result = predictor(test_day=test_day, train_day=train_day)
                    wr.writerow(result)
                except:
                    pass
                    # Ensure only valid days are tested on
