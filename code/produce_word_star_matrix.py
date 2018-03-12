import pandas as pd
import numpy as np
import re
from nltk.corpus import stopwords
from numpy import array
from numpy import append


#set the system not to print warnings
pd.options.mode.chained_assignment = None

#Read the data
yelp_30000 = pd.read_csv("train30000.csv")
text_30000 = yelp_30000.loc[:,"text"]

#Remove the noise
stops = set(stopwords.words("english"))
keep = {"all","do","don't","her","his","him","should", "shouldn't","not","aren't","couldn't","didn't",'didn','doesn',
        "doesn't","didn't","wouldn't","won't","won","shan't","hasn't","hadn't",
        "weren't","wasn't","mustn't","mightn't","weren't","very","couldn't",
        "isn't","haven't","haven't"}
for word in keep:
    stops.remove(word)

from nltk.stem.wordnet import WordNetLemmatizer
lem = WordNetLemmatizer()

def noise_remove(review, stops):
    #only keep characters ! and ?
    review = re.sub("\n", " ", review)
    review = re.sub("!", " !", review)
    review = re.sub("\?", " ?", review)
    review = re.sub("[^a-zA-Z!?]", " ", review)
    review = re.sub("[^a-zA-Z!?']", " ", review)
    #change to lower case
    review = review.lower()
    #split words like isn't to is not
    review = re.sub("n't"," not",review)
    review = re.sub("n'"," not",review)
    #remove the stop words
    review = review.split()
    useful_review = [word for word in review if not word in stops]
    #normalize the verb and noun
    useful_review = [lem.lemmatize(word, "v") for word in useful_review]
    return " ".join(useful_review)

for i in range(30000):
    text_30000[i] = noise_remove(text_30000[i],stops)
    
#produce the word dictionary
wordcount={}
for i in range(30000):
    text = text_30000[i].split()
    for j in range(len(text)):
        word = text[j]
        if word in ("not","no","never"):
            if (j+1) < len(text):
                word = text[j] + " " + text[j+1]
                j = j+1
        if word not in wordcount:
            wordcount[word] = 1
        else:
            wordcount[word] += 1

print("The length of dictionary is:",len(wordcount.keys()))


#cut down words with counts lower than 10
word_count_value = np.asarray(list(wordcount.values()))
word_count_key = np.asarray(list(wordcount.keys()))
word_count_key = word_count_key[word_count_value.astype(int)>=10]

wordcount2 = {}
for word in word_count_key:
    wordcount2[word] = wordcount[word]
wordcount = wordcount2


#count the conditional stars 

stars_30000 = yelp_30000.loc[:,"stars"]

words_stars = array([["first_line",0,0,0,0,0]])
k=0
for word in wordcount:
    new_line = [[word,0,0,0,0,0]]
    for i in range(30000):
        text = text_30000[i].split()
        for j in range(len(text)):
            word2 = text[j]
            if word2 in ("not","no","never"):
                if (j+1) < len(text):
                    word2 = text[j] + " " + text[j+1]
                    j = j + 1
            if word == word2:
                new_line[0][stars_30000[i]] += 1
    words_stars = append(words_stars,new_line,0)
    k=k+1
    print(k) 
words_stars = np.delete(words_stars,0,0)


#save and load the array 
import pickle

#save
f = open('words_stars3.pckl', 'wb')
pickle.dump(words_stars, f)
f.close()
