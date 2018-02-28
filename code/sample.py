import pandas as pd
from sklearn.model_selection import train_test_split
train_200 = pd.read_csv("train_data.csv")
train = train_200.sample(30000,random_state=123)
train.to_csv('train30000.csv',index= False)
