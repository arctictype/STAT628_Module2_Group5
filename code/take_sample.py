
# 取前70000行
yelp_30000 = pd.read_csv("train_data.csv",nrows=70000)
new_X = pd.read_csv("clean_test_dat.csv",nrows=70000)
new_X = new_X.drop(new_X.columns[0],1)

# 取前2000列，读入的矩阵一共有3000列
new_X = new_X.drop(new_X.columns[2000:3000],1)
Y = yelp_30000.loc[:,"stars"]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(new_X, Y, test_size=0.3, random_state=123)
