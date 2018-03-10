#python script to transform the test data into the matrix

#load the package
import sys
import pandas as pd
import numpy as np
import re

pd.options.mode.chained_assignment = None

test_data = pd.read_csv(sys.argv[1])
test_text = test_data.iloc[:,2]

from nltk.stem.wordnet import WordNetLemmatizer
lem = WordNetLemmatizer()

def noise_remove(review):
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
    #normalize the verb and noun
    useful_review = [lem.lemmatize(word, "v") for word in review]
    return " ".join(useful_review)

num_rows = len(test_text)

for i in range(num_rows):
    test_text[i] = noise_remove(test_text[i])

#load feature names
feature_names = pd.read_csv("feature_names.csv",index_col=None)
feature_names = np.asarray(feature_names.iloc[:,1])

#produce feature matrix
word_matrix = np.zeros((num_rows,200))

for i in range(num_rows):
    text = test_text[i].split()
    for j in range(200):
        if j < len(text):
            word = text[j]
            index = np.where(feature_names == word)[0]
            if index.__len__() == 0:
                word_matrix[i,j] = 0
            else:
                word_matrix[i,j] = index+1
        else:
            word_matrix[i,j] = 0

#save the result
word_matrix_df = pd.DataFrame(word_matrix)
word_matrix_df.to_csv(sys.argv[2])






