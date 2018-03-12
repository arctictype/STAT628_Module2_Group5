import pickle
import numpy as np
import pandas as pd
import re
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.corpus import stopwords
from numpy import array
from numpy import append
import sys

f = open('words_stars3.pckl', 'rb')
words_stars = pickle.load(f)
f.close()

#scale the conditional stars count
#(1) scale by overall star counts
words_stars[:,1] = words_stars[:,1].astype(int)/3.198
words_stars[:,2] = words_stars[:,2].astype(int)/2.938
words_stars[:,3] = words_stars[:,3].astype(int)/4.478
words_stars[:,4] = words_stars[:,4].astype(int)/8.489
words_stars[:,5] = words_stars[:,5].astype(int)/10.897

#(2) scale groupwisely
import math

for i in range(len(words_stars)):
    values = words_stars[i,1:6].astype(float)
    for j in range(len(values)):
        value = values[j]
        if(value>1):
            values[j] = math.log2(value)
    words_stars[i,1:6] = values


#Calculate the conditional variance
cond_variance = np.array([["first_line",0,0]])

for i in range(len(words_stars)):
    stars_distribution1 = np.array(words_stars[i][1:6].astype(float))
    stars_distribution2 = np.array([sum(words_stars[i][1:4].astype(float)),sum(words_stars[i][4:6].astype(float))])
    var1 = np.var(stars_distribution1)
    var2 = np.var(stars_distribution2)
    new_line = [[words_stars[i][0],var1,var2]]
    cond_variance = np.append(cond_variance,new_line,0)
    
cond_variance = np.delete(cond_variance,0,0)


#Get the 200 and print and save
temp = cond_variance[:,1].astype(float)
index = np.argsort(-temp)[range(8000)]
feature_names = cond_variance[index,0]
feature_values = cond_variance[index,1]

yelp_30000 = pd.read_csv(sys.argv[1])
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

for i in range(len(text_30000)):
    text_30000[i] = noise_remove(text_30000[i],stops)
    
#Produce the feature matrix
def initialize_matrix(value,row,col):
    matrix = []                                                      
    for i in range (0, row):     
        new = []                 
        for j in range (0, col):   
            new.append(value)      
        matrix.append(new)
    return matrix

a = yelp_30000.shape[0]
feature_matrix = initialize_matrix(0,a,8000)

feature_names1 = pd.DataFrame(feature_names)
feature_names2 = set(feature_names1[0])

feature_names1["num"] = list(range(8000))

for i in range(a):
    text = text_30000[i].split()
    for j in range(len(text)):
        word = text[j]
        if word in ("not","no","never"):
            if (j+1) < len(text):
                word = text[j] + " " + text[j+1]
                j = j + 1
        if word in feature_names2:
            index = feature_names1["num"][feature_names1[0]==word]
            j = index.get(index.keys()[0])
            feature_matrix[i][j] = feature_matrix[i][j] + 1


feature_matrix = pd.DataFrame(feature_matrix)
feature_matrix.to_csv(sys.argv[2])
