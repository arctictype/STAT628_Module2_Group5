import pandas as pd
import numpy as np
import re
import sys
from numpy import array
from numpy import append
from nltk.stem.wordnet import WordNetLemmatizer

#Read the data
yelp_30000 = pd.read_csv(sys.argv[1])

text_30000 = yelp_30000.text
print(len(text_30000))

pos_pc = pd.read_csv('pos.csv')
neg_pc = pd.read_csv('neg.csv')

con_pc = pd.concat([pos_pc,neg_pc])

feature_names = con_pc['V1']
feature_values = con_pc['V2']

print("next : produce feature matrix")

#Produce the feature matrix
def initialize_matrix(value,row,col):
    matrix = []                                                      
    for i in range (0, row):     
        new = []                 
        for j in range (0, col):   
            new.append(value)      
        matrix.append(new)
    return matrix


feature_matrix = initialize_matrix(0,len(text_30000),1670)

feature_names1 = pd.DataFrame(feature_names[0:1670])

feature_names1["num"] = list(range(1670))
print("next : begin the huge loop")
for i in range(len(text_30000)):
    text = text_30000[i].split()
    for j in range(len(text)):
        word = text[j]
        if word in ("not","no","never"):
            if (j+1) < len(text):
                word = text[j] + " " + text[j+1]
                j = j + 1
        if word in list(feature_names):
            index = feature_names1["num"][feature_names1['V1']==word]
            j = index.get(index.keys()[0])
            feature_matrix[i][j] = feature_matrix[i][j] + 1
    print(i)


feature_matrix = pd.DataFrame(feature_matrix)
feature_matrix.to_csv(sys.argv[2])
