import pandas as pd
import nltk
#nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer 
import time
import numpy as np
import os
import re
import langid
import seaborn as sns
import matplotlib.pyplot as plt
# first 200 lines of the train dataset
train_200 = pd.read_csv("first10000.csv")
def get_langid(dat_set):
    '''
    This function is used to return a numpy array which shows language kind of text
    '''
    lang_kind = []
    for i in range(len(dat_set)):
        lang_kind.append(langid.classify(dat_set["text"][i])[0])
    lang_id = np.array(lang_kind)
    return(lang_id)
lang_id = get_langid(train_200)
new_train = train_200.loc[(lang_id == 'en')]

stops = set(stopwords.words("english"))
porter_stemmer = PorterStemmer()
for i in new_train.index.values:
    t1 = re.sub("[^a-zA-Z]", " ",new_train["text"][i])
    # remove stopwords
    t2 = [w for w in t1.lower().split() if not w in stops]
    t3 = [porter_stemmer.stem(w) for w in t2]
    new_train["text"][i] = " ".join(t3)
# text_length is how many character the text have
new_train['text_length'] = new_train['text'].apply(len)

#g = sns.FacetGrid(data=new_train, col='stars')
#g.map(plt.hist, 'text_length', bins=50)
#plt.show()
X = new_train['text']
Y = new_train['stars']

from sklearn.feature_extraction.text import CountVectorizer
bow_transformer = CountVectorizer().fit(X)
#print(len(bow_transformer.vocabulary_))
new_X = bow_transformer.transform(X)
print('Shape of Sparse Matrix: ', new_X.shape)
print('Amount of Non-Zero occurrences: ', new_X.nnz)
# Percentage of non-zero values
density = (100.0 * new_X.nnz / (new_X.shape[0] * new_X.shape[1]))
print('Density: {}'.format((density)))

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(new_X, Y, test_size=0.3, random_state=101)

from sklearn.naive_bayes import MultinomialNB
nb = MultinomialNB()
nb.fit(X_train, y_train)

preds = nb.predict(X_test)

from sklearn.metrics import confusion_matrix, classification_report
print(confusion_matrix(y_test, preds))
print('\n')
print(classification_report(y_test, preds))
print(list(y_test))
print(preds)
sum = 0
for i in range(len(preds)):
    sum = sum + (list(y_test)[i]-preds[i])^2
print(sum)
print(len(preds))


############################# Here's the result for first 10000 rows of data #######################################
#Shape of Sparse Matrix:  (9848, 16611)
#Amount of Non-Zero occurrences:  456592
#Density: 0.27911583747392976
#[[180  54  28  27  19]
#[ 58  44  64  93  27]
#[ 28  27  93 223  58]
#[ 12  11  50 407 355]
#[  7   2  12 264 812]]
#
#
#             precision    recall  f1-score   support
#
#          1       0.63      0.58      0.61       308
#          2       0.32      0.15      0.21       286
#          3       0.38      0.22      0.28       429
#          4       0.40      0.49      0.44       835
#          5       0.64      0.74      0.69      1097
#
# avg / total       0.50      0.52      0.50      2955
# sqrt(711/2955) is the square-root of mean square error
