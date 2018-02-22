import pandas as pd
import nltk
from nltk.corpus import stopwords
import time

filename = 'train_data.csv'
t1 = time.time()
time.time() - t1
reviews = pd.read_csv('train_data.csv')

rev_200 = reviews.iloc[0:200,]
rev_200.to_csv("first200.csv", index = False)
