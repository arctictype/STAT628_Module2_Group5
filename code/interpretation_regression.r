#PCA
# PC2 value
da1 = read_csv("pos_neg2.csv")
# data matrix
da2 = read_csv("clean_train_29997.csv")
#da3 = read_csv("pc1score.csv")
# row data
da4 = read_csv("pos_neg1.csv")

# positive/ negative name list
posname = read_csv("pos.csv")
negname = read_csv("neg.csv")
p = as.data.frame(posname[,1])
n = as.data.frame(negname[,1])

#pc = cbind(rowSums(da1[-1]),rowSums(da3[-1]))
pc2 = rbind(as.data.frame(posname[,2]), as.data.frame(negname[,2]))

totalname = rbind(p,n)
# name list
wordname = c()
for (i in 1:dim(totalname)[1]){
  wordname[i] = totalname[i,1]
}

yelp = data.frame(stars = da2$stars, da4[,-1])
model1 = lm(stars~., data = yelp)
fitted0 = model1$fitted.values
index1 = fitted0<=1
index2 = fitted0>=5
fitted0[index1] = 1
fitted0[index2] = 5
rmse0 = sqrt(sum((fitted0-yelp$stars)^2)/30000)
rmse0
su = summary(model1)
su
#su2 = summary(model3)$coefficients
co = su$coefficients
co = as.data.frame(co)
co = cbind(co, pc2)
colnames(co) = c("Estimate", "Std.Error", "t_value", "pvalue")
rownames(co) = wordname1
# select pvalue < 0.05
result = co[co$pvalue<0.05,]
positive = result[result$Estimate>=0,]
negative = result[result$Estimate<0,]

layout(1)
plot(model1, which = 1)
plot(model1, which = 2)
plot(model1, which = 3)
plot(model1, which = 4)
plot(model1, which = 5)
