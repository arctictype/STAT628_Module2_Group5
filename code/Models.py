import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt
# first 200 lines of the train dataset
train_200 = pd.read_csv("clean200.csv")

X = train_200['text']
Y = train_200['stars']

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
bow_transformer1 = CountVectorizer().fit(X)
#bow_transformer2 = TfidfVectorizer(ngram_range=(1,2)).fit(X)
#print(len(bow_transformer.vocabulary_))
new_X = bow_transformer1.transform(X)
#time = pd.to_numeric(train_200['date'].str[:4], errors='coerce')
print('Shape of Sparse Matrix: ', new_X.shape)
print('Amount of Non-Zero occurrences: ', new_X.nnz)

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
