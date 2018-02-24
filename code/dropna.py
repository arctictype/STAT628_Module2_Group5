train_200 = pd.read_csv("clean_train_dat.csv")
train1 = train_200.dropna(subset=['text'])
