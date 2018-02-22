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
import operator
# first 200 lines of the train dataset
train_200 = pd.read_csv("first200.csv")
stops = set(stopwords.words("english"))
porter_stemmer = PorterStemmer()
for i in range(len(train_200)):
    t1 = re.sub("[^a-zA-Z]", " ",train_200["text"][i])
    # remove stopwords
    t2 = [w for w in t1.lower().split() if not w in stops]
    t3 = [porter_stemmer.stem(w) for w in t2]
    train_200["text"][i] = " ".join(t3)

# get a dictionary
def get_dict(data): # e.g. data = train_200["text"]
    wordsbox = []
    for line in data:
        wordsbox.extend(line.split())
    return collections.Counter(wordsbox)

vocabulary = get_dict(train_200["text"])
# e.g. get the first ten mostly used words
sorted(vocabulary.items(),key=lambda k:k[1], reverse=True)[:11] 