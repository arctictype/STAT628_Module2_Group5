import pandas as pd
import nltk
from nltk.corpus import stopwords
import time

filename = 'train_data.csv'
t1 = time.time()
time.time() - t1
reviews = pd.read_csv('train_data.csv')
