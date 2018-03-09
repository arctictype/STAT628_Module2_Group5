#library packages
library(readr)
library(nnet)

#read the data
yelp = read_csv("train30000.csv")
train_x = read_csv("Test04.csv")
train_x = train_x[,-1]
test_x = read_csv("Test04_10000.csv")
test_x = test_x[,-1]
yelp_10000 = read_csv("train_data.csv",n_max=10000)
test_y = yelp_10000$stars
y = yelp$stars


n=600
train_x_small = train_x[,1:n]
test_x_small = test_x[,1:n]
#linear model
lm1 = lm(y~.,data=train_x_small)


#measure2
fitted0 = lm1$fitted.values
index1 = fitted0<=1
index2 = fitted0>=5
fitted0[index1] = 1
fitted0[index2] = 5
rmse0 = sqrt(sum((fitted0-y)^2)/30000)
rmse0

fitted1 = predict(lm1,test_x_small)
index1 = fitted1<=1
index2 = fitted1>=5
fitted1[index1] = 1
fitted1[index2] = 5
rmse1 = sqrt(sum((fitted1-test_y)^2)/10000)
rmse1




#logistics model

lg1 = multinom(y~.,data=train_x_small,MaxNWts=10010,maxit=100)
fitted2 = as.numeric(predict(lg1,data=train_x_small))
rmse2 = sqrt(sum((fitted2-y)^2)/30000)
rmse2

fitted3 = as.numeric(predict(lg1,data=test_x_small[,1:n]))
rmse3 = sqrt(sum((fitted3[1:10000]-test_y)^2)/10000)
rmse3


