#package for data cleaning
import pandas as pd
import numpy as np
import re
from nltk.corpus import stopwords

#package for DNN model
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers.embeddings import Embedding
from keras.layers import Conv1D
from keras.layers import MaxPooling1D
from keras.layers import GRU


from keras.layers import Dense,Input,LSTM,Bidirectional,Activation,Conv1D,GRU
from keras.layers import Dropout,Embedding,GlobalMaxPooling1D, MaxPooling1D, Add, Flatten

from keras import initializers, regularizers, constraints, optimizers, layers, callbacks
from keras.callbacks import EarlyStopping,ModelCheckpoint
from keras.models import Model
from keras.optimizers import Adam


#set the system not to print warnings
pd.options.mode.chained_assignment = None

#read the data and transform them
yelp_100000 = pd.read_csv("train_data.csv",skiprows=410000,nrows=30000)
text_100000 = yelp_100000.iloc[:,2]
yelp_10000 = pd.read_csv("train_data.csv",skiprows=400000,nrows=10000)
text_10000 = yelp_10000.iloc[:,2]

#Remove the noise
from nltk.stem.wordnet import WordNetLemmatizer
lem = WordNetLemmatizer()

def noise_remove(review):
    #only keep characters ! and ?
    review = re.sub("\n", " ", review)
    review = re.sub("!", " !", review)
    review = re.sub("\?", " ?", review)
    review = re.sub("[^a-zA-Z!?]", " ", review)
    review = re.sub("[^a-zA-Z!?']", " ", review)
    review = re.sub("n't"," not",review)
    review = re.sub("n'"," not",review)
    #change to lower case
    review = review.lower()
    #split words like isn't to is not
    #remove the stop words
    review = review.split()
    #normalize the verb and noun
    useful_review = [lem.lemmatize(word, "v") for word in review]
    return " ".join(useful_review)

for i in range(30000):
    text_100000[i] = noise_remove(text_100000[i])
    print(i)
    
for i in range(10000):
    text_10000[i] = noise_remove(text_10000[i])
    print(i)
    
 
wordcount={}
for i in range(100000):
    text = text_100000[i].split()
    for word in text:
        if word not in wordcount:
            wordcount[word] = 1
        else:
            wordcount[word] += 1

print("The length of dictionary is:",len(wordcount.keys()))

word_count_value = np.asarray(list(wordcount.values()))
word_count_key = np.asarray(list(wordcount.keys()))
index = np.argsort(-word_count_value)[range(30000)]
feature_names = word_count_key[index]


#Get the 5000 and print and save
top_words = 30000
max_review_length = 200

#data_matrix

word_matrix = np.zeros((100000,max_review_length))

#save and load
#save1 = pd.DataFrame(word_matrix)
#save1.to_csv("word_matrix1.csv")
#word_matrix = pd.read_csv("word_matrix1.csv")
#word_matrix = word_matrix.iloc[:,1:201]
#word_matrix = np.asarray(word_matrix).astype(int)

for i in range(100000):
    text = text_100000[i].split()
    for j in range(max_review_length):
        if j < len(text):
            word = text[j]
            index = np.where(feature_names == word)[0]
            if index.__len__() == 0:
                word_matrix[i,j] = 0
            else:
                word_matrix[i,j] = index+1
        else:
            word_matrix[i,j] = 0
    print(i)        
            
word_matrix2 = np.zeros((10000,max_review_length))

#save and load
#save2 = pd.DataFrame(word_matrix2)
#save2.to_csv("word_matrix2.csv")
#word_matrix2 = pd.read_csv("word_matrix2.csv")
#word_matrix2 = word_matrix2.iloc[:,1:201]
#word_matrix2 = np.asarray(word_matrix2).astype(int)

for i in range(10000):
    text = text_10000[i].split()
    for j in range(max_review_length):
        if j < len(text):
            word = text[j]
            index = np.where(feature_names == word)[0]
            if index.__len__() == 0:
                word_matrix2[i,j] = 0
            else:
                word_matrix2[i,j] = index+1
        else:
            word_matrix2[i,j] = 0
    print(i)             

#set train and test
X_train = word_matrix.astype(int)
X_test = word_matrix2.astype(int)
y_train0 = np.asarray(yelp_100000.iloc[:,0])
y_test0 = np.asarray(yelp_10000.iloc[:,0])

y_train = np.zeros((100000,5))
y_test = np.zeros((10000,5))

for i in range(100000):
    y_train[i,y_train0[i]-1] = 1

for i in range(10000):
    y_test[i,y_test0[i]-1] = 1


# create the model
#Model1
inputs = Input(shape=(X_train.shape[1], ))
x = Embedding(30001,128)(inputs)
x = LSTM(128, dropout=0.5, recurrent_dropout=0.5)(x)
output = Dense(5, activation="sigmoid")(x)
model = Model(inputs, output)
model.compile(loss='mse',optimizer='adam',metrics=['accuracy'])
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=4, batch_size=128)


# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))

prediction = model.predict(X_test,verbose=0)

reuslt1 = np.argmax(prediction, axis=1)+1
rmse1 = np.sqrt(np.sum(np.square(reuslt1-y_test0))/10000)
    
reuslt2 = prediction[:,0] + 2*prediction[:,1] + 3*prediction[:,2] + 4*prediction[:,3] + 5*prediction[:,4]
index1 = reuslt2>5
reuslt2[index1] = 5
rmse2 = np.sqrt(np.sum(np.square(reuslt2-y_test0))/10000)

#save the model
model.save("model1.h5")

###########################################
test_data = pd.read_csv("final_test.csv")
test_data = test_data.iloc[:,1:201]

np.count_nonzero(test_data==0)/(200*1016665)
np.count_nonzero(X_train==0)/(200*100000)

final_prediction = model.predict(test_data,verbose=0)

final_result = final_prediction[:,0] + 2*final_prediction[:,1] + 3*final_prediction[:,2] + 4*final_prediction[:,3] + 5*final_prediction[:,4]
index1 = final_result>5
final_result[index1] = 5











