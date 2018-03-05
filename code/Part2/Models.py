import pandas as pd
import numpy as np

yelp_30000 = pd.read_csv("train30000.csv")
new_X = pd.read_csv("feature_matrix.csv")
new_X = new_X.drop(new_X.columns[0],1)
Y = yelp_30000.loc[:,"stars"]

print('Shape of Sparse Matrix: ', new_X.shape)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(new_X, Y, test_size=0.3, random_state=101)

########################### Here's Bayes ################################

from sklearn.naive_bayes import MultinomialNB
nb = MultinomialNB()
nb.fit(X_train, y_train)

preds = nb.predict(X_test)

from sklearn.metrics import confusion_matrix, classification_report, mean_squared_error
print(confusion_matrix(y_test, preds))
print('\n')
print(classification_report(y_test, preds))
print(mean_squared_error(y_test,preds))

# 10-fold cross-validation
from sklearn.model_selection import cross_val_predict
preds = cross_val_predict(nb, new_X, Y, cv=10)
print(mean_squared_error(preds,Y))
########################### Here's SVM ########################
print("Here's SVM")
from sklearn import svm
clf = svm.SVC(kernel='linear',class_weight='balanced', C=1).fit(X_train, y_train)
preds = clf.predict(X_test)
from sklearn.metrics import confusion_matrix, classification_report, mean_squared_error
print(confusion_matrix(y_test, preds))
print('\n')
print(classification_report(y_test, preds))
print(mean_squared_error(y_test,preds))

# 10-fold cross-validation
from sklearn.model_selection import cross_val_predict
preds = cross_val_predict(clf, new_X, Y, cv=10)
print(mean_squared_error(preds,Y))
