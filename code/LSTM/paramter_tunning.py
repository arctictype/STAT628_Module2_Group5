#package for data cleaning
import pandas as pd
import numpy as np


#package for RNN model
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers.embeddings import Embedding
from keras.layers import Conv1D
from keras.layers import MaxPooling1D
from keras.layers import GRU
from keras.layers import Input
from keras.models import Model


#set the system not to print warnings
pd.options.mode.chained_assignment = None


#read the data and transform them
yelp_all = pd.read_csv("train_data.csv",skiprows=410000,nrows=10000)
yelp_10000 = pd.read_csv("train_data.csv",skiprows=400000,nrows=5000)

#data_matrix
max_review_length = 200
word_matrix = pd.read_csv("word_matrix1.csv")
word_matrix = word_matrix.iloc[0:10000,1:201]

word_matrix2 = pd.read_csv("word_matrix2.csv")
word_matrix2 = word_matrix2.iloc[0:5000,1:201]
      

#set train and test
X_train = word_matrix.astype(int)
X_test = word_matrix2.astype(int)
y_train0 = np.asarray(yelp_all.iloc[0:10000,0])
y_test0 = np.asarray(yelp_10000.iloc[0:5000,0])

y_train = np.zeros((10000,5))
y_test = np.zeros((5000,5))

for i in range(10000):
    y_train[i,y_train0[i]-1] = 1

for i in range(5000):
    y_test[i,y_test0[i]-1] = 1


# create the model
#Model1
inputs = Input(shape=(X_train.shape[1], ))
x = Embedding(30001,16)(inputs)
x = LSTM(16,dropout=0.2, recurrent_dropout=0.2,implementation=1)(x)
output = Dense(5, activation="sigmoid")(x)
model = Model(inputs, output)
model.compile(loss='mse',optimizer='adam',metrics=['acc'])
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=2, batch_size=128)


# record the result
prediction = model.predict(X_test,verbose=0)

reuslt1 = prediction[:,0] + 2*prediction[:,1] + 3*prediction[:,2] + 4*prediction[:,3] + 5*prediction[:,4]
index1 = reuslt1>5
reuslt1[index1] = 5
rmse1 = np.sqrt(np.sum(np.square(reuslt1-y_test0))/5000)

prediction = model.predict(X_train,verbose=0)

reuslt2 = prediction[:,0] + 2*prediction[:,1] + 3*prediction[:,2] + 4*prediction[:,3] + 5*prediction[:,4]
index2 = reuslt2>5
reuslt2[index2] = 5
rmse2 = np.sqrt(np.sum(np.square(reuslt1-y_test0))/10000)



embed = [2,4,8,16,32,64,128,256,512]
embed_record = np.zeros((9,2))

for i in range(9):
    embed_unit = embed[i]
    inputs = Input(shape=(X_train.shape[1], ))
    x = Embedding(30001,8)(inputs)
    x = LSTM(embed_unit,dropout=0.2, recurrent_dropout=0.2,implementation=1)(x)
    output = Dense(5, activation="sigmoid")(x)
    model = Model(inputs, output)
    model.compile(loss='mse',optimizer='adam',metrics=['acc'])
    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=2, batch_size=128)

    prediction = model.predict(X_test,verbose=0)

    reuslt1 = prediction[:,0] + 2*prediction[:,1] + 3*prediction[:,2] + 4*prediction[:,3] + 5*prediction[:,4]
    index1 = reuslt1>5
    reuslt1[index1] = 5
    rmse1 = np.sqrt(np.sum(np.square(reuslt1-y_test0))/5000)

    prediction = model.predict(X_train,verbose=0)

    reuslt2 = prediction[:,0] + 2*prediction[:,1] + 3*prediction[:,2] + 4*prediction[:,3] + 5*prediction[:,4]
    index2 = reuslt2>5
    reuslt2[index2] = 5
    rmse2 = np.sqrt(np.sum(np.square(reuslt1-y_test0))/10000)
    lstm_record[i,0] = rmse2
    lstm_record[i,1] = rmse1
    print(i)




