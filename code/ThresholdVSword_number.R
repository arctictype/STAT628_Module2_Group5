dat = read.csv('word_count_value.csv')
a = sort(unique(table(dat$X0)),decreasing = FALSE)

count = NULL
for (i in seq(1,100,by=1)){
  count=c(count,sum(dat$X0>i))
}
plot(seq(1,100,by=1),count,type='b',cex.axis = 1.5, cex.lab = 1.5,xlab = "Threshold",ylab = "# {words | word's count > threshold}")
abline(v = 10,col="red")
text(cex = 2,20,20000,"Threshold = 10")
