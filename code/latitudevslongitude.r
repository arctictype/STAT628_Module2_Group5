library(readr)
city = read_csv("train30000.csv")
hist(city$longitude,freq = FALSE)
hist(city$latitude,freq = FALSE)

class0 = which(city$longitude < -100)
class1 = which(city$longitude > -100 & city$longitude < -50)
class2 = which(city$longitude > -50 & city$longitude < 50)
class3 = which(city$longitude > 100)

mean(city$stars[class0])
mean(city$stars[class1])
mean(city$stars[class2])
mean(city$stars[class3])
barplot()
group = rep(0,30000)
group[class0] = 1
group[class1] = 2
group[class2] = 3
group[class3] = 1
library(ggplot2)
dat = data.frame(group = group,longitude=city$longitude,Avg_star = city$stars)
ggplot(dat,aes(x=group,fill = factor(Avg_star))) + geom_histogram(binwidth=.5, position="dodge")

par(fig=c(0,0.8,0,0.8))
plot(city$longitude, city$latitude, cex.axis = 1.5,cex.lab = 1.5,xlab="Longitude",
     ylab="Latitude",pch=16,col=group)
legend('topright',text.font = 2,legend=c('West North America','East North America','Europe'),pch=c(16,16,16),col=c(1,2,3))
par(fig=c(0,0.8,0.55,1), new=TRUE)
boxplot(city$longitude, horizontal=TRUE, axes=FALSE)
par(fig=c(0.65,1,0,0.8),new=TRUE)
boxplot(city$latitude, axes=FALSE)
