import pandas as pd
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer 
import time
import numpy as np
import os
import re
import langid
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.cross_validation import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.svm import LinearSVC
import jieba
import operator
import collections
import seaborn as sns
import matplotlib.pyplot as plt
# first 200 lines of the train dataset
#train_200 = pd.read_csv("first200.csv")

def data_clean(dataset):
    ''' change to lowercase, stemming, remove punctuation and stopwords
    '''
    stops = set(stopwords.words("english"))
    porter_stemmer = PorterStemmer()
    for i in range(len(dataset)):
        t1 = re.sub("[^a-zA-Z]", " ",dataset["text"][i])
        t2 = [w for w in t1.lower().split() if not w in stops]
        t3 = [porter_stemmer.stem(w) for w in t2]
        dataset.iloc[i, 2] = " ".join(t3)
    return dataset
#train_200.iloc[0,2]

def data_combine(traindata, testdata):
    '''
    combine the traindata and testdata together
    to get words array easily
    '''
    testdata1 = testdata.drop(["Id"], axis=1)
    col_name = testdata1.columns.tolist()
    col_name.insert(0, "stars")
    combined_data = traindata.append(testdata1)
    return combined_data.reindex(columns=col_name)



# get a dictionary
def get_dict(data): # e.g. data = train_200["text"]
    wordsbox = []
    for line in data:
        wordsbox.extend(line.split())
    return collections.Counter(wordsbox)



def get_array(data, K):
    '''
    This function gives an array of the top K frequent words.
    @ data: cleaned text reviews (e.g. train_200["text"])
    @ K: the top K frequent words
    '''
    indexlist = []
    cv = CountVectorizer()
    model_fit = cv.fit_transform(data)
    modelarray = model_fit.toarray() # get the array of whole words
    nameslist = cv.get_feature_names() # get the names of columns
    vocabulary = get_dict(data)
    freqlist = sorted(vocabulary.items(),key=lambda k:k[1], reverse=True)[:K]
    for i in range(len(freqlist)):
        word = freqlist[i][0]
        indexlist.append(nameslist.index(word))
    return modelarray[:,indexlist]

#vocabulary = get_dict(train_200["text"])
# e.g. get the first five mostly used words
#sorted(vocabulary.items(),key=lambda k:k[1], reverse=True)[:5] 
# get its words frequency array
#array_30 = get_array(train_200["text"], 30)   
#array_50 = get_array(train_200["text"], 50)
#array_100 = get_array(train_200["text"], 100)
#array_200 = get_array(train_200["text"], 200)

#regr = linear_model.LinearRegression()
#regr.fit(array_200, train_200["stars"])
# predict train data
#test_pre = regr.predict(array_200)
#np.sqrt(mean_squared_error(train_200["stars"], test_pre))
# K=30, RMSE = 0.97851722704892674
# K=50, RMSE = 0.91642845185644428
# K=100, RMSE = 0.66892633059728757
# K=200, RMSE = 4.3097350608929815e-14

train_cleaned = pd.read_csv("clean_train_dat.csv.csv")
train = train_cleaned.dropna(subset=["text"])

#TF-IDF
cv = CountVectorizer()
transformer = TfidfTransformer()
cv_fit = cv.fit_transform(train["text"])
tfidf = transformer.fit_transform(cv_fit)
weight = tfidf.toarray()

# divide train data
x_train, x_test, y_train, y_test = train_test_split(cv_fit.toarray(), train["stars"], test_size = 0.3, random_state=13)

#logistic
#x_train = transformer.fit_transform(cv_fit)
#x_test = transformer.transform(cv_fit)
classifier = LogisticRegression(multi_class= "multinomial", solver="sag")
classifier.fit(x_train, y_train)
#precisions = cross_val_score(classifier, x_train, train_200["stars"], cv=5, scoring='precision')
preds = classifier.predict(x_test)
np.sqrt(mean_squared_error(y_train, preds))
# RMSE = 0.69735213486444569

#linear_model
regr = linear_model.LinearRegression()
regr.fit(x_train, y_train)
# predict train data
test_pre = regr.predict(x_test)
np.sqrt(mean_squared_error(y_train, test_pre))
regr.coef_
# cv_fit: 0.0077660028726112473
# tfidf: 0.0076305680216021835
#v1 = get_dict(train1w["text"])
#len(v1)
# toarray = (10000, 19188) len_of_whole_words = 19206