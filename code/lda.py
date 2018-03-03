"""
=======================================================================================
Topic extraction with Non-negative Matrix Factorization and Latent Dirichlet Allocation
=======================================================================================

This is an example of applying :class:`sklearn.decomposition.NMF` and
:class:`sklearn.decomposition.LatentDirichletAllocation` on a corpus
of documents and extract additive models of the topic structure of the
corpus.  The output is a list of topics, each represented as a list of
terms (weights are not shown).

Non-negative Matrix Factorization is applied with two different objective
functions: the Frobenius norm, and the generalized Kullback-Leibler divergence.
The latter is equivalent to Probabilistic Latent Semantic Indexing.

The default parameters (n_samples / n_features / n_components) should make
the example runnable in a couple of tens of seconds. You can try to
increase the dimensions of the problem, but be aware that the time
complexity is polynomial in NMF. In LDA, the time complexity is
proportional to (n_samples * iterations).

"""

# Author: Olivier Grisel <olivier.grisel@ensta.org>
#         Lars Buitinck
#         Chyi-Kwei Yau <chyikwei.yau@gmail.com>
# License: BSD 3 clause

from __future__ import print_function
from time import time
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation
import numpy as np
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
import re
from nltk.corpus import stopwords



n_samples = 200
n_features = 100
n_components = 10
n_top_words = 20


def print_top_words(model, feature_names, n_top_words):
    for topic_idx, topic in enumerate(model.components_):
        message = "Topic #%d: " % topic_idx
        message += " ".join([feature_names[i]
                             for i in topic.argsort()[:-n_top_words - 1:-1]])
        print(message)
    print()


# Load the 20 newsgroups dataset and vectorize it. We use a few heuristics
# to filter out useless terms early on: the posts are stripped of headers,
# footers and quoted replies, and common English words, words occurring in
# only one document or in at least 95% of the documents are removed.


# train_200 = pd.read_csv("clean200.csv")
# train1 = train_200.dropna(subset=['text'])
# new_train = train1



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
    review = re.sub("\n", " ", review)
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
''''
wordcount={}
for i in range(0,29999):
    for word in text_30000[i].split():
        if word not in wordcount:
            wordcount[word] = 1
        else:
            wordcount[word] += 1

print("The length of dictionary is:",len(wordcount.keys()))
'''

X = text_30000
Y = yelp_30000['stars']

# Use tf (raw term count) features for LDA.
print("Extracting tf features for LDA...")
tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2)
tf = tf_vectorizer.fit_transform(X)

print("Fitting LDA models with tf features, "
      "n_samples=%d and n_features=%d..."
      % (n_samples, n_features))
lda = LatentDirichletAllocation(n_components=n_components, max_iter=5,
                                learning_method='online',
                                learning_offset=50.,
                                random_state=0)

model = lda.fit(tf)
pc = pd.DataFrame(np.transpose(model.components_))
pc.to_csv('pc.csv')
print("\nTopics in LDA model:")
tf_feature_names = tf_vectorizer.get_feature_names()
print_top_words(lda, tf_feature_names, n_top_words)
