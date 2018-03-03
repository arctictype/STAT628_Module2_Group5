import pandas as pd
import numpy as np
import nltk
import re
from nltk.corpus import stopwords

#set the system not to print warnings
pd.options.mode.chained_assignment = None

#Read the data
yelp_30000 = pd.read_csv("train30000.csv")
text_30000 = yelp_30000.loc[:,"text"]

#Remove the noise
stops = set(stopwords.words("english"))
keep = {"all","do","don't","her","his","him","should", "shouldn't","not"}
for word in keep:
    stops.remove(word)

from nltk.stem.wordnet import WordNetLemmatizer
lem = WordNetLemmatizer()

def noise_remove(review, stops):
    #only keep characters ! and ?
    review = re.sub("[^a-zA-Z!?]", " ", review)
    #change to lower case
    review = review.lower().split()
    #remove the stop words
    useful_review = [word for word in review if not word in stops]
    #normalize the verb and noun
    useful_review = [lem.lemmatize(word, "v") for word in useful_review]
    return " ".join(useful_review)

for i in range(0,29999):
    text_30000[i] = noise_remove(text_30000[i],stops)

#Count all the words
wordcount={}
for i in range(0,29999):
    for word in text_30000[i].split():
        if word not in wordcount:
            wordcount[word] = 1
        else:
            wordcount[word] += 1

print("The length of dictionary is:",len(wordcount.keys()))