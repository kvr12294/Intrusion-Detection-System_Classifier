import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle

import sklearn
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn import svm
from datetime import datetime
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from os import system
import csv


def readFile(filename):
    """
    Reading the file and choosing attributes
    :param filename: the file to be read
    :return: the data read
    """
    file = open(filename)
    csv_f = csv.reader(file)
    skip_first_line = 0
    classified = []
    row_num = 0
    classes = {}
    num = 1
    exclude_features = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,24,25,30,31,36,38]
    num_attr = 42 - len(exclude_features)
    if filename == "Final_dataset.csv":
        data = [[0 for col in range(num_attr)] for row in range(75000)]
    else:
        data = [[0 for col in range(num_attr)] for row in range(28000)]
    for read in csv_f:
        row = []
        if skip_first_line == 0:
            skip_first_line += 1
        else:
            for i in range(0,len(read)):
                if i not in exclude_features:
                    row.append(read[i])
            r = [float(i) for i in row[0:(num_attr - 1)]]
            data[row_num] = r[0:(num_attr - 1)]
            if row[(num_attr - 1)] in classes:
                classified.append(classes[row[(num_attr - 1)]])
            else:
                classes[row[(num_attr - 1)]] = num
                classified.append(classes[row[(num_attr - 1)]])
                num += 1
            row_num += 1
    # print(classes)
    file.close()
    return data, classified


def Support_VM(X,Y,X_test,Y_test):
    """
    This function creates a Support Vector for the given Dataset
    :param X: The data without target variable
    :param Y: The target variable
    :param X_test: The test data without target variable
    :param Y_test: The test data target variable
    """
    print("Constructing Support Vector Model")
    svm_model = OneVsRestClassifier(svm.SVC(kernel='linear', probability=True, max_iter=2))
    svm_model = svm_model.fit(X,Y)
    print("Finished constructing model")
    print("Testing model...")

    prediction = svm_model.decision_function(X_test)

    Y_test = label_binarize(Y_test, classes=[1, 2, 3, 4, 5, 6])
    prediction = label_binarize(prediction, classes=[1, 2, 3, 4, 5, 6])
    print("Accuracy is: ",end = "")
    print(str(accuracy_score(Y_test,prediction)*100) + "%\n")


def Random_Forest(X,Y,X_test,Y_test):
    """
    This function creates a Random Forest classification model for the given Dataset with estimators = 10
    :param X: The data without target variable
    :param Y: The target variable
    :param X_test: The test data without target variable
    :param Y_test: The test data target variable
    """
    print("\n------Random_Forest------")
    print("Constructing Random Forest")
    model = OneVsRestClassifier(RandomForestClassifier(n_estimators=10))
    model = model.fit(X,Y)
    print("Finished constructing model")
    print("Testing model...")

    prediction = model.predict(X_test)

    Y_test = label_binarize(Y_test, classes=[1, 2, 3, 4, 5, 6])
    prediction = label_binarize(prediction, classes=[1, 2, 3, 4, 5, 6])
    print("\nAccuracy is: ",end = "")
    print(str(accuracy_score(Y_test,prediction)*100) + "%")

    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(6):
        fpr[i], tpr[i], _ = roc_curve(Y_test[:, i], prediction[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    for i in range(6):
        plt.figure()
        plt.plot(fpr[i],tpr[i])
        plt.xlabel("FPR")
        plt.ylabel("TPR")
        plt.title("ROC for model")
        plt.show()


def Logistic_Regression(X,Y,X_test,Y_test):
    """
    This function creates a Logistic Regression model for the given Dataset
    :param X: The data without target variable
    :param Y: The target variable
    :param X_test: The test data without target variable
    :param Y_test: The test data target variable
    """
    print("\n------Logistic_Regression------")
    print("Constructing Logistic Regression")
    model = OneVsRestClassifier(LogisticRegression())
    model = model.fit(X,Y)
    print("Finished constructing model")
    print("Testing model...")

    prediction = model.predict(X_test)

    Y_test = label_binarize(Y_test, classes=[1, 2, 3, 4, 5, 6])
    prediction = label_binarize(prediction, classes=[1, 2, 3, 4, 5, 6])
    print("\nAccuracy is: ",end = "")
    print(str(accuracy_score(Y_test,prediction)*100) + "%")

    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(6):
        fpr[i], tpr[i], _ = roc_curve(Y_test[:, i], prediction[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    for i in range(6):
        plt.figure()
        plt.plot(fpr[i],tpr[i])
        plt.xlabel("FPR")
        plt.ylabel("TPR")
        plt.title("ROC for model")
        plt.show()


def Naive_Bayes(X,Y,X_test,Y_test):
    """
    This function creates a Naive Bayes classification model for the given Dataset
    :param X: The data without target variable
    :param Y: The target variable
    :param X_test: The test data without target variable
    :param Y_test: The test data target variable
    """
    print("\n------Naive Bayes------")
    print("Constructing Naive Bayes")
    model = OneVsRestClassifier(GaussianNB())
    model = model.fit(X,Y)
    print("Finished constructing model")
    print("Testing model...")

    prediction = model.predict(X_test)

    Y_test = label_binarize(Y_test, classes=[1, 2, 3, 4, 5, 6])
    prediction = label_binarize(prediction, classes=[1, 2, 3, 4, 5, 6])
    print("\nAccuracy is: ",end = "")
    print(str(accuracy_score(Y_test,prediction)*100) + "%")




def KNN(X,Y,X_test,Y_test):
    """
    This function creates a KNN classification model for the given Dataset with k = 17
    :param X: The data without target variable
    :param Y: The target variable
    :param X_test: The test data without target variable
    :param Y_test: The test data target variable
    """
    print("\n------K Nearest Neighbor------")
    print("Constructing KNN")
    model = OneVsRestClassifier(KNeighborsClassifier(n_neighbors=17))
    model = model.fit(X,Y)
    print("Finished constructing model")
    print("Testing model...")

    prediction = model.predict(X_test)

    Y_test = label_binarize(Y_test, classes=[1, 2, 3, 4, 5, 6])
    prediction = label_binarize(prediction, classes=[1, 2, 3, 4, 5, 6])
    print("\nAccuracy is: ",end = "")
    print(str(accuracy_score(Y_test,prediction)*100) + "%")

    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(6):
        fpr[i], tpr[i], _ = roc_curve(Y_test[:, i], prediction[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    for i in range(6):
        plt.figure()
        plt.plot(fpr[i],tpr[i])
        plt.xlabel("FPR")
        plt.ylabel("TPR")
        plt.title("ROC for model")
        plt.show()




def Decision_tree(X,Y,X_test,Y_test):
    """
    This function creates a Decision Tree model for the given Dataset
    :param X: The data without target variable
    :param Y: The target variable
    :param X_test: The test data without target variable
    :param Y_test: The test data target variable
    :return:
    """
    print("\n------Decision_Tree------")
    print("Constructing Decision tree")
    tree_model = OneVsRestClassifier(DecisionTreeClassifier())
    tree_model = tree_model.fit(X,Y)
    print("Finished constructing")
    prediction = tree_model.predict(X_test)

    Y_test = label_binarize(Y_test, classes=[1, 2, 3, 4, 5, 6]) #binarizing the data
    prediction = label_binarize(prediction, classes=[1, 2, 3, 4, 5, 6])
    print("\nAccuracy is: ",end = "")
    print(str(accuracy_score(Y_test,prediction)*100) + "%")

    # fpr = dict()
    # tpr = dict()
    # roc_auc = dict()
    # for i in range(6):
    #     fpr[i], tpr[i], _ = roc_curve(Y_test[:, i], prediction[:, i])
    #     roc_auc[i] = auc(fpr[i], tpr[i])
    # for i in range(6):
    #     plt.figure()
    #     plt.plot(fpr[i],tpr[i])
    #     plt.xlabel("FPR")
    #     plt.ylabel("TPR")
    #     plt.title("ROC for model")
    #     plt.show()

    print("Testing model...")
    feature_names=["ipsweep","normal","neptune","smurf","portsweep","satan"]
    with open('tree.dot', 'w') as dotfile:
        export_graphviz(DecisionTreeClassifier().fit(X,Y),dotfile)
    system("dot -Tpng D:.dot -o D:/dtree2.png")


def main():
    """
    This is the main function of the class
    """
    print("Reading files..")
    X,Y = readFile("Final_dataset.csv")
    X_test,Y_test = readFile("Final_test_dataset.csv")
    print("Finished reading files")
    print(str(datetime.now()))
    Decision_tree(X,Y,X_test,Y_test)
    print(str(datetime.now()))
    KNN(X,Y,X_test,Y_test)
    print(str(datetime.now()))
    Naive_Bayes(X,Y,X_test,Y_test)
    print(str(datetime.now()))
    Logistic_Regression(X,Y,X_test,Y_test)
    print(str(datetime.now()))
    Random_Forest(X,Y,X_test,Y_test)
    print(str(datetime.now()))
    # Support_VM(X,Y,X_test,Y_test)
    # print(str(datetime.now()))


if __name__ == '__main__':
    main()
