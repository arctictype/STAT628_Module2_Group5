#import the packages
library(dplyr)
library(ggplot2)
library(readr)

words = read.csv("feature1.csv")
freqs = read.csv("feature2.csv")
words = words[,2]
freqs = freqs[,2]


freqs = freqs*10

wordcloud(words, freqs, random.order=FALSE,colors=brewer.pal(8,'Dark2'))

wordcloud(words,freqs,c(8,.3),2)
