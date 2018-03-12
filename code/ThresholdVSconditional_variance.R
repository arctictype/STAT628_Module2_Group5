dat = read.csv('featrue_variance_value.csv')
a = sort(unique(table(dat$X0)),decreasing = FALSE)

count = NULL
for (i in seq(0,6,by=0.01)){
  count=c(count,sum(dat$X0>i))
}
plot(seq(0,6,by=0.01),count,type='b',cex.axis = 1.5, cex.lab = 1.5,xlab = "Threshold",ylab = "# {words | word's count > threshold}")
abline(v = 0.5,col="red")
text(cex = 2,1.5,6000,"Threshold = 0.5")
