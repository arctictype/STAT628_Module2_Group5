import pandas as pd
reviews = pd.read_csv('testval_data.csv')
total_len = int(reviews.shape[0]) # 1546379
num_file = int(total_len/1000) # 1546
for i in range((num_file)): # 0-999: 1000-1999: 1545000 - 1545999 rows
    lower = i*1000
    higher = (i+1)*1000
    part = reviews.iloc[lower:higher,]
    part.to_csv('./test_data/data_'+format(i,'04d')+'.csv')
if (total_len % 1000)>0:
    part = reviews.iloc[(num_file*1000):total_len,]
    part.to_csv('./test_data/data_'+format(num_file,'04d')+'.csv')
