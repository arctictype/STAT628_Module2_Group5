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
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.svm import LinearSVC
import operator
import collections
import seaborn as sns
import matplotlib.pyplot as plt

def data_clean(dataset):
    ''' 
    change to lowercase, stemming, remove punctuation and stopwords
    '''
    stops = set(stopwords.words("english"))
    porter_stemmer = PorterStemmer()
    for i in range(len(dataset)):
        t1 = re.sub("[^a-zA-Z]", " ",dataset["text"][i])
        t2 = [w for w in t1.lower().split() if not w in stops]
        t3 = [porter_stemmer.stem(w) for w in t2]
        dataset.iloc[i, 2] = " ".join(t3)
    return dataset


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

#vocabulary = get_dict(train_200["text"])
# e.g. get the first ten mostly used words
#sorted(vocabulary.items(),key=lambda k:k[1], reverse=True)[:11] 


train_data = pd.read_csv("train_data.csv") #(1546379, 8)
testval_data = pd.read_csv("testval_data.csv") #(1016664, 8)
d1 = data_clean(train_data)
d1.to_csv("train_cleaned.csv", index= False)
d2 = data_clean(testval_data)
d2.to_csv("testval_cleaned.csv", index= False)
com_data = data_combine(d1, d2)
com_data.to_csv("wholedata_cleaned.csv", index= False)
# 0:1546378 are train data columns, 1546379:2563043 are testval data columns



