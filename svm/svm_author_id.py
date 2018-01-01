#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()




#########################################################
### your code goes here ###

#########################################################
### create classifier
from sklearn.svm import SVC
# clf = SVC(kernel="linear")
clf = SVC(kernel="rbf", C=10000)

# features_train = features_train[:len(features_train)/100] 
# labels_train = labels_train[:len(labels_train)/100] 

### fit the classifier on the training features and labels
t0 = time()
clf.fit(features_train, labels_train)
print "training time:", round(time()-t0, 3), "s"

### use the trained classifier to predict labels for the test features
t0 = time()
pred = clf.predict(features_test)
print "predict time:", round(time()-t0, 3), "s"

print "element 10", pred[10]
print "element 26", pred[26]
print "element 50", pred[50]

print 'Chris emails ', sum(1 for item in pred if item==(1))

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(pred, labels_test)

print 'accuracy', accuracy