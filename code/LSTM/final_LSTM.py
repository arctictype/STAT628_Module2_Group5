#package for data cleaning
import pandas as pd
import numpy as np


#package for DNN model
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
yelp_all = pd.read_csv("train_data.csv")
yelp_10000 = pd.read_csv("train_data.csv",skiprows=400000,nrows=10000)

#data_matrix
max_review_length = 200
word_matrix = pd.read_csv("final_train.csv")
word_matrix = word_matrix.iloc[:,1:201]

word_matrix2 = pd.read_csv("word_matrix2.csv")
word_matrix2 = word_matrix2.iloc[:,1:201]
      

#set train and test
X_train = word_matrix.astype(int)
X_test = word_matrix2.astype(int)
y_train0 = np.asarray(yelp_all.iloc[:,0])
y_test0 = np.asarray(yelp_10000.iloc[:,0])

y_train = np.zeros((1546379,5))
y_test = np.zeros((10000,5))

for i in range(1546379):
    y_train[i,y_train0[i]-1] = 1

for i in range(10000):
    y_test[i,y_test0[i]-1] = 1


# create the model
#Model1
inputs = Input(shape=(X_train.shape[1], ))
x = Embedding(30001,128)(inputs)
x = LSTM(128,dropout=0.2, recurrent_dropout=0.2)(x)
output = Dense(5, activation="sigmoid")(x)
model = Model(inputs, output)
model.compile(loss='mse',optimizer='adam',metrics=['accuracy'])
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=4, batch_size=128)

#save the model
model.save("model4.h5")


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



###########################################
test_data = pd.read_csv("final_test.csv")
test_data = test_data.iloc[:,1:201]

np.count_nonzero(test_data==0)/(200*1016665)
np.count_nonzero(X_train==0)/(200*100000)

final_prediction = model.predict(test_data,verbose=0)

final_result = final_prediction[:,0] + 2*final_prediction[:,1] + 3*final_prediction[:,2] + 4*final_prediction[:,3] + 5*final_prediction[:,4]
index1 = final_result>5
final_result[index1] = 5












