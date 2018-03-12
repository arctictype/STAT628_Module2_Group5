library(readr)
city = read_csv("train30000.csv")

dat = data.frame(star = city$stars,lat = city$latitude,longi = city$longitude,text_len=city$text_len,year = factor(substring(city$date,1,4)))
pairs(dat)

hist(city$longitude,freq = FALSE)
hist(city$latitude,freq = FALSE)

class0 = which(city$longitude < -100 | city$longitude > 100)
class1 = which(city$longitude > -100 & city$longitude < -50)
class2 = which(city$longitude > -50 & city$longitude < 50)

mean(city$stars[class0])
mean(city$stars[class1])
mean(city$stars[class2])
mean(city$stars[class3])

group = rep(0,30000)
group[class0] = 1
group[class1] = 2
group[class2] = 3
group[class3] = 1

par(fig=c(0,0.8,0,0.8))
plot(city$longitude, city$latitude, cex.axis = 1.5,cex.lab = 1.5,xlab="Longitude",
     ylab="Latitude",pch=16,col=group)
legend('topright',text.font = 2,legend=c('West North America','East North America','Europe'),pch=c(16,16,16),col=c(1,2,3))
par(fig=c(0,0.8,0.55,1), new=TRUE)
boxplot(city$longitude, horizontal=TRUE, axes=FALSE)
par(fig=c(0.65,1,0,0.8),new=TRUE)
boxplot(city$latitude, axes=FALSE)
