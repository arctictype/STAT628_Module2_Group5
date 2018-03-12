import pandas as pd
import numpy as np

yelp_30000 = pd.read_csv("train30000.csv")
new_X1 = pd.read_csv("new_train30000.csv")
new_X1 = new_X1.drop(new_X1.columns[0],1)
Y = yelp_30000.loc[:,"stars"]
samplesize = [10,50,100,500,1000,2000,3000,4000,5000]

for i in samplesize:
    new_X = new_X1.drop(new_X1.columns[i:8000],1)
    print("shape:"+str(new_X.shape))
    ############  svm ####################
    from sklearn import svm
    clf = svm.SVC(kernel='linear',class_weight='balanced',C=1)

    from sklearn.model_selection import cross_val_predict
    preds = cross_val_predict(clf, new_X, Y, cv=10)
    from sklearn.metrics import mean_squared_error
    print(np.sqrt(mean_squared_error(preds,Y)))
    ########### random forest ################
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import GridSearchCV
    rf = RandomForestRegressor(n_estimators=180, criterion="mse", max_depth=66,
                               min_samples_split=35, min_samples_leaf=1, min_weight_fraction_leaf=0.0,
                               max_features="auto", max_leaf_nodes=None, min_impurity_decrease=0.0,
                               min_impurity_split=None, bootstrap=True, oob_score=True, n_jobs=1,
                               random_state=101, verbose=0, warm_start=False)
    from sklearn.model_selection import cross_val_predict
    preds = cross_val_predict(rf, new_X, Y, cv=10)
    from sklearn.metrics import mean_squared_error
    print(np.sqrt(mean_squared_error(preds,Y)))
    ########### linear ############
    from sklearn import linear_model
    from sklearn.linear_model import LinearRegression
    regr = linear_model.LinearRegression()
    from sklearn.model_selection import cross_val_predict
    preds = cross_val_predict(regr, new_X, Y, cv=10)
    from sklearn.metrics import mean_squared_error
    print(np.sqrt(mean_squared_error(preds,Y)))
    ########### Logistic ###############
    from sklearn.linear_model.logistic import LogisticRegression
    logi = LogisticRegression(multi_class= "multinomial", solver="sag")
    from sklearn.model_selection import cross_val_predict
    preds = cross_val_predict(logi, new_X, Y, cv=10)
    from sklearn.metrics import mean_squared_error
    print(np.sqrt(mean_squared_error(preds,Y)))
