dat = read.csv('word.csv',stringsAsFactors = FALSE)
colnames(dat) <- c('id','words','1star','2star','3star','4star','5star')
prin_dat = dat[,c('1star','2star','3star','4star','5star')]
prin_comp = princomp(scale(prin_dat))
summary(prin_comp)
prin_comp$loadings

# Cumulative plot
cum_prop = c(0.9194589,0.96753132,0.98308960,0.99312478,1.000000000)
x = c(1,2,3,4,5)
plot(x,cum_prop,type = 'b',lwd = 2,cex.axis = 1.5, cex.lab = 1.5,xlab = "Number of components",ylab = "Cumulative Propotion")

PC2 = prin_comp$scores[,2]
PC1 = prin_comp$scores[,1]
pos_csv = cbind(dat$words[which(PC2 < (-0.5))],PC2[which(PC2 < (-0.5))],PC1[which(PC2 < (-0.5))])
neg_csv = cbind(dat$words[which(PC2 > 0.5)],PC2[which(PC2 > 0.5)],PC1[which(PC2 < (-0.5))])
# Save them to do regression
write.csv(pos_csv,'pos.csv')
write.csv(neg_csv,'neg.csv')

# PLOT PC2 distribution
hist(prin_comp$scores[,2],col = c(2,2,2,2,0,0,1,1,1,1,1,1),cex.axis = 1.5, cex.lab = 1.5,cex.main = 1.5,xlab = "Component 2",main = 'Distribution of PC2')
