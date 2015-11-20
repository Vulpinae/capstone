setwd("~/Coursera/10_Capstone/Data")

Needed <- c("tm", "SnowballCC", "RColorBrewer", "ggplot2", "wordcloud", "biclust", "cluster", "igraph", "fpc")   
install.packages(Needed, dependencies=TRUE)   

install.packages("RWeka")   

install.packages("Rcampdf", repos = "http://datacube.wu.ac.at/", type = "source")   cname <- file.path("~", "Desktop", "texts")   


library(e1071);library(tm);library(RWeka);library(openNLP);library(caret)

# working directory
setwd("~/Coursera/10_Capstone/Data")

# used packages
library(RJSONIO)
library(RCurl)
library(plyr)


# The following code checks if the right files are in your working directory
# If not, it downloads them and unzips them

#---------------------------------------------------
# Verify input file exists.  If not, then get it.  
#---------------------------------------------------
sourcefile <- "https://d396qusza40orc.cloudfront.net/dsscapstone/dataset/yelp_dataset_challenge_academic_dataset.zip"
zipfile    <- "yelp_dataset_challenge_academic_dataset.zip"
jsonfile    <- "yelp_academic_dataset_business.json"
if(!file.exists(zipfile)) {download.file(sourcefile, zipfile, mode="wb", method="curl")}
if(!file.exists(jsonfile)) {unzip(zipfile)}  #unzip creates .json file


#read data
json_file1 <-("yelp_dataset_challenge_academic_dataset/yelp_academic_dataset_business.json")
json_file2 <-("yelp_dataset_challenge_academic_dataset/yelp_academic_dataset_checkin.json")
json_file3 <-("yelp_dataset_challenge_academic_dataset/yelp_academic_dataset_tip.json")
json_file4 <-("yelp_dataset_challenge_academic_dataset/yelp_academic_dataset_user.json")
json_file5 <-("yelp_dataset_challenge_academic_dataset/yelp_academic_dataset_review.json")

# turn it into a proper array by separating each object with a "," and
# wrapping that up in an array with "[]"'s.

business <- fromJSON(sprintf("[%s]", paste(readLines(json_file1), collapse=",")), flatten=TRUE)
saveRDS(business, "business.RDS")

# For categorie variable -> Separate breakdown list and original data frame into different objects
df <- business
breakdown_list <- df$categories
df$categories <- NULL

#Loop over df and accumulate results
parsed_df <- data.frame()
for(i in 1:nrow(df)){
        right_df <-  breakdown_list[[i]]
        if (length(right_df)==0) {right_df<-"NA"}
        # right_df <- rename(right_df, replace=c("name" = report_raw$report$elements$id[2]))
        temp <- cbind(df[i,],right_df, row.names = NULL)
        parsed_df <- rbind(parsed_df, temp)
}
names(parsed_df)[names(parsed_df) == 'right_df' <- 'categorie'
                 
business_parsed<-parsed_df
saveRDS(business_parsed,"business_parsed.RDS")
                 
                 
checkin <- fromJSON(sprintf("[%s]", paste(readLines(json_file2), collapse=",")), flatten=TRUE)
saveRDS(checkin,"checkin.RDS")
                 
tip <- fromJSON(sprintf("[%s]", paste(readLines(json_file3), collapse=",")), flatten=TRUE)
saveRDS(tip,"tip.RDS")
                 
user <- fromJSON(sprintf("[%s]", paste(readLines(json_file4), collapse=",")), flatten=TRUE)
saveRDS(user,"user.RDS")
                 
review <- fromJSON(sprintf("[%s]", paste(readLines(json_file5), collapse=",")), flatten=TRUE)
saveRDS(review,"review.RDS")
                 
                 
# read data

business<-readRDS("business.RDS")
business_parsed<-readRDS("business_parsed.RDS")
checkin<-readRDS("checkin.RDS")
tip<-readRDS("tip.RDS")
user<-readRDS("user.RDS")
review<-readRDS("review.RDS")

names(business_parsed[105]) <- "categorie"

cats<-count(business_parsed$right_df)
names(cats)<-c("category", "freq")
cats2<-cats[order(cats$freq,decreasing=TRUE),]

# only vegas restaurants
restaurant<-unique(business_parsed[business_parsed$right_df=="Restaurants",c(1,4,6,9)])
restaurant_vegas<-restaurant[restaurant$city=="Las Vegas",]
rr<-merge(review,restaurant_vegas, by.x="business_id", by.y="business_id")
saveRDS(rr,"rr.RDS")

# **Load the R package for text mining and then load your texts into R.**
rrv<-readRDS("rrv_10k.RDS")
order<-order(rrv$review_id)
rrv<-rrv[order,]
saveRDS(rrv,"rrv_10k.RDS")

set.seed(111)

######################################
######## Bag of words ################
######################################

review_source <- VectorSource(rrv$text)
docs<-Corpus(review_source)


stopwords_raw <- readLines(system.file("stopwords", "english.dat",package = "tm"))
non_stopwords<-c("isn't", "aren't", "wasn't", "weren't", "hasn't", "haven't", "hadn't",
                 "doesn't", "don't", "didn't", "won't", "wouldn't", "shan't", "shouldn't", 
                 "can't", "cannot", "couldn't", "mustn't", "few", "most", "nor", "not", "only", "too", "very")
stopwords_new <- setdiff(stopwords_raw, non_stopwords) 

## Preprocessing      
docs <- tm_map(docs, removePunctuation)   # *Removing punctuation:*    
docs <- tm_map(docs, removeNumbers)      # *Removing numbers:*    
docs <- tm_map(docs, tolower)   # *Converting to lowercase:*    
docs <- tm_map(docs, removeWords, stopwords_new)   # *Removing "stopwords" 
docs <- tm_map(docs, stemDocument)   # *Removing common word endings* (e.g., "ing", "es")   
docs <- tm_map(docs, stripWhitespace)   # *Stripping whitespace   
docs <- tm_map(docs, PlainTextDocument)   
## *This is the end of the preprocessing stage.*   


### Stage the Data      
#dtm.matrix <- as.matrix(DocumentTermMatrix(docs))   

dtm<-DocumentTermMatrix(docs)
saveRDS(dtm,"dtm.RDS")
FreqTerms100<-findFreqTerms(dtm, lowfreq=1090)  
FreqTerms150<-findFreqTerms(dtm, lowfreq=815)  
FreqTerms250<-findFreqTerms(dtm, lowfreq=532)  
FreqTerms500<-findFreqTerms(dtm, lowfreq=266)  
dtm100<-dtm[,FreqTerms100]
dtm150<-dtm[,FreqTerms150]
dtm250<-dtm[,FreqTerms250]
dtm500<-dtm[,FreqTerms500]
dtms100.matrix<-as.matrix(dtm100)
dtms150.matrix<-as.matrix(dtm150)
dtms250.matrix<-as.matrix(dtm250)
dtms500.matrix<-as.matrix(dtm500)

# tf-dif
dtms100.matrix<-dtms100.matrix*log(nrow(dtms100.matrix)/colSums(dtms100.matrix>0))
dtms150.matrix<-dtms150.matrix*log(nrow(dtms150.matrix)/colSums(dtms150.matrix>0))
dtms250.matrix<-dtms250.matrix*log(nrow(dtms250.matrix)/colSums(dtms250.matrix>0))
dtms500.matrix<-dtms500.matrix*log(nrow(dtms500.matrix)/colSums(dtms500.matrix>0))

saveRDS(dtms100.matrix,"dtms100_matrix.RDS")
dtms100.matrix<-readRDS("dtms100_matrix.RDS")
saveRDS(dtms150.matrix,"dtms150_matrix.RDS")
dtms150.matrix<-readRDS("dtms150_matrix.RDS")
saveRDS(dtms250.matrix,"dtms250_matrix.RDS")
dtms250.matrix<-readRDS("dtms250_matrix.RDS")
saveRDS(dtms500.matrix,"dtms500_matrix.RDS")
dtms500.matrix<-readRDS("dtms500_matrix.RDS")





########################################################
#################      bigrams        ##################
########################################################

dtm.generate <- function(string, ng){
        
        # tutorial on rweka - http://tm.r-forge.r-project.org/faq.html
        corpus <- Corpus(VectorSource(string)) # create corpus for TM processing
        corpus <- tm_map(corpus, content_transformer(tolower))
        corpus <- tm_map(corpus, removeWords, stopwords_new) 
        corpus <- tm_map(corpus, removeNumbers) 
        corpus <- tm_map(corpus, removePunctuation)
        corpus <- tm_map(corpus, stripWhitespace)
        corpus <- tm_map(corpus, stemDocument) 
        corpus <- tm_map(corpus, PlainTextDocument)
        options(mc.cores=1) # http://stackoverflow.com/questions/17703553/bigrams-instead-of-single-words-in-termdocument-matrix-using-r-and-rweka/20251039#20251039
        BigramTokenizer <- function(x) NGramTokenizer(x, Weka_control(min = ng, max = ng)) # create n-grams
        dtm <- DocumentTermMatrix(corpus, control = list(tokenize = BigramTokenizer)) # create tdm from n-grams
        dtm
}

dtm2 <- dtm.generate(rrv$text, 2)

#remove sparse terms (15% remaining)
FreqTerms100<-findFreqTerms(dtm2, lowfreq=139)  
FreqTerms150<-findFreqTerms(dtm2, lowfreq=110)  
FreqTerms250<-findFreqTerms(dtm2, lowfreq=81)  
FreqTerms500<-findFreqTerms(dtm2, lowfreq=57)  
dtmb100<-dtm2[,FreqTerms100]
dtmb150<-dtm2[,FreqTerms150]
dtmb250<-dtm2[,FreqTerms250]
dtmb500<-dtm2[,FreqTerms500]
dtmb100.matrix<-as.matrix(dtmb100)
dtmb150.matrix<-as.matrix(dtmb150)
dtmb250.matrix<-as.matrix(dtmb250)
dtmb500.matrix<-as.matrix(dtmb500)

# tf-dif
dtmb100.matrix<-dtmb100.matrix*log(nrow(dtmb100.matrix)/colSums(dtmb100.matrix>0))
dtmb150.matrix<-dtmb150.matrix*log(nrow(dtmb150.matrix)/colSums(dtmb150.matrix>0))
dtmb250.matrix<-dtmb250.matrix*log(nrow(dtmb250.matrix)/colSums(dtmb250.matrix>0))
dtmb500.matrix<-dtmb500.matrix*log(nrow(dtmb500.matrix)/colSums(dtmb500.matrix>0))

saveRDS(dtmb100.matrix,"dtmb100_matrix.RDS")
dtmb100.matrix<-readRDS("dtmb100_matrix.RDS")
saveRDS(dtmb150.matrix,"dtmb150_matrix.RDS")
dtmb150.matrix<-readRDS("dtmb150_matrix.RDS")
saveRDS(dtmb250.matrix,"dtmb250_matrix.RDS")
dtmb250.matrix<-readRDS("dtmb250_matrix.RDS")
saveRDS(dtmb500.matrix,"dtmb500_matrix.RDS")
dtmb500.matrix<-readRDS("dtmb500_matrix.RDS")








############################################
######    Adjectives by POS tagging   ###### 
############################################

## Need sentence and word token annotations.
sent_token_annotator <- Maxent_Sent_Token_Annotator()
word_token_annotator <- Maxent_Word_Token_Annotator()
pos_tag_annotator <- Maxent_POS_Tag_Annotator()

adject<-matrix(nrow=nrow(rrv))

for (i in 1:nrow(rrv)) 
{
        s<-as.String(rrv[i,6])
        a2 <- annotate(s, list(sent_token_annotator, word_token_annotator))
        a3 <- annotate(s, pos_tag_annotator, a2)
        ## Determine the distribution of POS tags for word tokens.
        a3w <- subset(a3, type == "word")
        tags <- sapply(a3w$features, `[[`, "POS")
        adj<-tags=="JJ"
        ## Extract token/POS pairs (all of them): easy.
        magweg1<-s[a3w]
        if (length(magweg1)==1) {magweg2 <- "NA"} else {magweg2<-magweg1[adj]}
        adject[i,1]<-magweg2[1]
        for (j in 2:length(magweg2)) 
        { adject[i,1] = paste(adject[i,1], magweg2[j])
        }
        if (i %% 100 == 0) {print(i)}
}


#docs <- Corpus(DirSource(cname))   
review_source <- VectorSource(adject)
docsadj<-Corpus(review_source)

## Preprocessing      
docsadj <- tm_map(docsadj, removePunctuation)   # *Removing punctuation:*    
docsadj <- tm_map(docsadj, removeNumbers)      # *Removing numbers:*    
docsadj <- tm_map(docsadj, tolower)   # *Converting to lowercase:*    
docsadj <- tm_map(docsadj, removeWords, stopwords_new)   # *Removing "stopwords" 
docsadj <- tm_map(docsadj, stemDocument)   # *Removing common word endings* (e.g., "ing", "es")   
docsadj <- tm_map(docsadj, stripWhitespace)   # *Stripping whitespace   
docsadj <- tm_map(docsadj, PlainTextDocument)   
## *This is the end of the preprocessing stage.*   

### Stage the Data      
dtmadj <- DocumentTermMatrix(docsadj)   

#remove sparse terms (15% remaining)
FreqTerms100<-findFreqTerms(dtmadj, lowfreq=203)  
FreqTerms150<-findFreqTerms(dtmadj, lowfreq=126)  
FreqTerms250<-findFreqTerms(dtmadj, lowfreq=67)  
FreqTerms500<-findFreqTerms(dtmadj, lowfreq=24)  
dtma100<-dtmadj[,FreqTerms100]
dtma150<-dtmadj[,FreqTerms150]
dtma250<-dtmadj[,FreqTerms250]
dtma500<-dtmadj[,FreqTerms500]
dtma100.matrix<-as.matrix(dtma100)
dtma150.matrix<-as.matrix(dtma150)
dtma250.matrix<-as.matrix(dtma250)
dtma500.matrix<-as.matrix(dtma500)

# tf-dif
dtma100.matrix<-dtma100.matrix*log(nrow(dtma100.matrix)/colSums(dtma100.matrix>0))
dtma150.matrix<-dtma150.matrix*log(nrow(dtma150.matrix)/colSums(dtma150.matrix>0))
dtma250.matrix<-dtma250.matrix*log(nrow(dtma250.matrix)/colSums(dtma250.matrix>0))
dtma500.matrix<-dtma500.matrix*log(nrow(dtma500.matrix)/colSums(dtma500.matrix>0))

#dtmadj2.matrix<-matrix(nrow=nrow(dtmadj.matrix), ncol=ncol(dtmadj.matrix))
#for (i in 1:ncol(dtmadj.matrix)) {
#        dtmadj2.matrix[,i]<-as.numeric(dtmadj.matrix[,i])
#}

saveRDS(dtma100.matrix,"dtma100_matrix.RDS")
dtma100.matrix<-readRDS("dtma100_matrix.RDS")
saveRDS(dtma150.matrix,"dtma150_matrix.RDS")
dtma150.matrix<-readRDS("dtma150_matrix.RDS")
saveRDS(dtma250.matrix,"dtma250_matrix.RDS")
dtma250.matrix<-readRDS("dtma250_matrix.RDS")
saveRDS(dtma500.matrix,"dtma500_matrix.RDS")
dtma500.matrix<-readRDS("dtma500_matrix.RDS")


#saveRDS(dtmadj2.matrix,"dtmadj2_matrix.RDS")
#dtmadj2.matrix<-readRDS("dtmadj2_matrix.RDS")


# Make model
set.seed(1)


# create datapartition
inTrain <- createDataPartition(y=rrv$stars, p=.7, list=FALSE)
TrainStars<-as.factor(rrv[inTrain,4])
RealStars<-as.factor(rrv[-inTrain,4])
RealText<-rrv[-inTrain,6]

# prepare y values 
y1<-ifelse(rrv[inTrain,4]==1,1,0)
y2<-ifelse(rrv[inTrain,4]==2,1,0)
y3<-ifelse(rrv[inTrain,4]==3,1,0)
y4<-ifelse(rrv[inTrain,4]==4,1,0)
y5<-ifelse(rrv[inTrain,4]==5,1,0)




#single
training_s100 <- dtms100.matrix[inTrain,]
testing_s100 <- dtms100.matrix[-inTrain,]
training_s150 <- dtms150.matrix[inTrain,]
testing_s150 <- dtms150.matrix[-inTrain,]
training_s250 <- dtms250.matrix[inTrain,]
testing_s250 <- dtms250.matrix[-inTrain,]
training_s500 <- dtms500.matrix[inTrain,]
testing_s500 <- dtms500.matrix[-inTrain,]

#remove high correlations
cor <- findCorrelation(cor(training_s100))
ncor<-length(cor)
if (ncor > 0) {training_s100 <- training_s100[,-cor]} 
if (ncor > 0) {testing_s100 <- testing_s100[,-cor]}

cor <- findCorrelation(cor(training_s150))
ncor<-length(cor)
if (ncor > 0) {training_s150 <- training_s150[,-cor]} 
if (ncor > 0) {testing_s150 <- testing_s150[,-cor]}

cor <- findCorrelation(cor(training_s250))
ncor<-length(cor)
if (ncor > 0) {training_s250 <- training_s250[,-cor]} 
if (ncor > 0) {testing_s250 <- testing_s250[,-cor]}

cor <- findCorrelation(cor(training_s500))
ncor<-length(cor)
if (ncor > 0) {training_s500 <- training_s500[,-cor]} 
if (ncor > 0) {testing_s500 <- testing_s500[,-cor]}


# bigrams
training_b100 <- dtmb100.matrix[inTrain,]
testing_b100 <- dtmb100.matrix[-inTrain,]
training_b150 <- dtmb150.matrix[inTrain,]
testing_b150 <- dtmb150.matrix[-inTrain,]
training_b250 <- dtmb250.matrix[inTrain,]
testing_b250 <- dtmb250.matrix[-inTrain,]
training_b500 <- dtmb500.matrix[inTrain,]
testing_b500 <- dtmb500.matrix[-inTrain,]

#remove high correlations
cor <- findCorrelation(cor(training_b100))
ncor<-length(cor)
if (ncor > 0) {training_b100 <- training_b100[,-cor]} 
if (ncor > 0) {testing_b100 <- testing_b100[,-cor]}

cor <- findCorrelation(cor(training_b150))
ncor<-length(cor)
if (ncor > 0) {training_b150 <- training_b150[,-cor]} 
if (ncor > 0) {testing_b150 <- testing_b150[,-cor]}

cor <- findCorrelation(cor(training_b250))
ncor<-length(cor)
if (ncor > 0) {training_b250 <- training_b250[,-cor]} 
if (ncor > 0) {testing_b250 <- testing_b250[,-cor]}

cor <- findCorrelation(cor(training_b500))
ncor<-length(cor)
if (ncor > 0) {training_b500 <- training_b500[,-cor]} 
if (ncor > 0) {testing_b500 <- testing_b500[,-cor]}


# adjectives
training_a100 <- dtma100.matrix[inTrain,]
testing_a100 <- dtma100.matrix[-inTrain,]
training_a150 <- dtma150.matrix[inTrain,]
testing_a150 <- dtma150.matrix[-inTrain,]
training_a250 <- dtma250.matrix[inTrain,]
testing_a250 <- dtma250.matrix[-inTrain,]
training_a500 <- dtma500.matrix[inTrain,]
testing_a500 <- dtma500.matrix[-inTrain,]

#remove high correlations
cor <- findCorrelation(cor(training_a100))
ncor<-length(cor)
if (ncor > 0) {training_a100 <- training_a100[,-cor]} 
if (ncor > 0) {testing_a100 <- testing_a100[,-cor]}

cor <- findCorrelation(cor(training_a150))
ncor<-length(cor)
if (ncor > 0) {training_a150 <- training_a150[,-cor]} 
if (ncor > 0) {testing_a150 <- testing_a150[,-cor]}

cor <- findCorrelation(cor(training_a250))
ncor<-length(cor)
if (ncor > 0) {training_a250 <- training_a250[,-cor]} 
if (ncor > 0) {testing_a250 <- testing_a250[,-cor]}

cor <- findCorrelation(cor(training_a500))
ncor<-length(cor)
if (ncor > 0) {training_a500 <- training_a500[,-cor]} 
if (ncor > 0) {testing_a500 <- testing_a500[,-cor]}







# regression models 
# simple 100 features

train1_s100<-as.data.frame(cbind(training_s100,y1))
model1_s100 <- glm(y1 ~ .,family=binomial(),data = train1_s100, control = list(maxit = 100))
pred1_s100 <- as.matrix(fitted(model1_s100))

train2_s100<-as.data.frame(cbind(training_s100,y2))
model2_s100 <- glm(y2 ~ .,family=binomial(),data = train2_s100, control = list(maxit = 100))
pred2_s100 <- as.matrix(fitted(model2_s100))

train3_s100<-as.data.frame(cbind(training_s100,y3))
model3_s100 <- glm(y3 ~ .,family=binomial(),data = train3_s100, control = list(maxit = 100))
pred3_s100 <- as.matrix(fitted(model3_s100))

train4_s100<-as.data.frame(cbind(training_s100,y4))
model4_s100 <- glm(y4 ~ .,family=binomial(),data = train4_s100, control = list(maxit = 100))
pred4_s100 <- as.matrix(fitted(model4_s100))

train5_s100<-as.data.frame(cbind(training_s100,y5))
model5_s100 <- glm(y5 ~ .,family=binomial(),data = train5_s100, control = list(maxit = 100))
pred5_s100 <- as.matrix(fitted(model5_s100))

# determine prediction on test set
totpred_s100<-as.data.frame(cbind(TrainStars,pred1_s100,pred2_s100,pred3_s100,pred4_s100,pred5_s100))
totpred_s100[,"max"]<-apply(totpred_s100[,2:6],1,max)
PredStars1_s100<-ifelse(totpred_s100[,2]==totpred_s100[,7],1,0)
PredStars2_s100<-ifelse(totpred_s100[,3]==totpred_s100[,7],2,0)
PredStars3_s100<-ifelse(totpred_s100[,4]==totpred_s100[,7],3,0)
PredStars4_s100<-ifelse(totpred_s100[,5]==totpred_s100[,7],4,0)
PredStars5_s100<-ifelse(totpred_s100[,6]==totpred_s100[,7],5,0)
totpred_s100$PredStars<-PredStars1_s100+PredStars2_s100+PredStars3_s100+PredStars4_s100+PredStars5_s100

# confusion matrix on test set
CM_pred_s100<-table(totpred_s100$PredStars,totpred_s100$TrainStars)
acc_pred_s100<-sum(ifelse(totpred_s100[,1]==totpred_s100[,8],1,0))/nrow(totpred_s100)
saveRDS(CM_pred_s100, "CM_pred_s100.RDS")
saveRDS(acc_pred_s100, "acc_pred_s100.RDS")

# predict on validation set
val1_s100<-as.matrix(predict(model1_s100,newdata=as.data.frame(testing_s100), type="response"))
val2_s100<-as.matrix(predict(model2_s100,newdata=as.data.frame(testing_s100), type="response"))
val3_s100<-as.matrix(predict(model3_s100,newdata=as.data.frame(testing_s100), type="response"))
val4_s100<-as.matrix(predict(model4_s100,newdata=as.data.frame(testing_s100), type="response"))
val5_s100<-as.matrix(predict(model5_s100,newdata=as.data.frame(testing_s100), type="response"))

# classify on validation set
valpred_s100<-as.data.frame(cbind(RealStars,val1_s100,val2_s100,val3_s100,val4_s100,val5_s100))
valpred_s100[,"max"]<-apply(valpred_s100[,2:6],1,max)
ValStars1_s100<-ifelse(valpred_s100[,2]==valpred_s100[,7],1,0)
ValStars2_s100<-ifelse(valpred_s100[,3]==valpred_s100[,7],2,0)
ValStars3_s100<-ifelse(valpred_s100[,4]==valpred_s100[,7],3,0)
ValStars4_s100<-ifelse(valpred_s100[,5]==valpred_s100[,7],4,0)
ValStars5_s100<-ifelse(valpred_s100[,6]==valpred_s100[,7],5,0)
valpred_s100$ValStars<-ValStars1_s100+ValStars2_s100+ValStars3_s100+ValStars4_s100+ValStars5_s100
CM_val_s100<-table(valpred_s100$ValStars,valpred_s100$RealStars)
acc_val_s100<-sum(ifelse(valpred_s100[,1]==valpred_s100[,8],1,0))/nrow(valpred_s100)
saveRDS(CM_val_s100, "CM_val_s100.RDS")
saveRDS(acc_val_s100, "acc_val_s100.RDS")


# simple 150 features

train1_s150<-as.data.frame(cbind(training_s150,y1))
model1_s150 <- glm(y1 ~ .,family=binomial(),data = train1_s150, control = list(maxit = 100))
pred1_s150 <- as.matrix(fitted(model1_s150))

train2_s150<-as.data.frame(cbind(training_s150,y2))
model2_s150 <- glm(y2 ~ .,family=binomial(),data = train2_s150, control = list(maxit = 100))
pred2_s150 <- as.matrix(fitted(model2_s150))

train3_s150<-as.data.frame(cbind(training_s150,y3))
model3_s150 <- glm(y3 ~ .,family=binomial(),data = train3_s150, control = list(maxit = 100))
pred3_s150 <- as.matrix(fitted(model3_s150))

train4_s150<-as.data.frame(cbind(training_s150,y4))
model4_s150 <- glm(y4 ~ .,family=binomial(),data = train4_s150, control = list(maxit = 100))
pred4_s150 <- as.matrix(fitted(model4_s150))

train5_s150<-as.data.frame(cbind(training_s150,y5))
model5_s150 <- glm(y5 ~ .,family=binomial(),data = train5_s150, control = list(maxit = 100))
pred5_s150 <- as.matrix(fitted(model5_s150))

# determine prediction on test set
totpred_s150<-as.data.frame(cbind(TrainStars,pred1_s150,pred2_s150,pred3_s150,pred4_s150,pred5_s150))
totpred_s150[,"max"]<-apply(totpred_s150[,2:6],1,max)
PredStars1_s150<-ifelse(totpred_s150[,2]==totpred_s150[,7],1,0)
PredStars2_s150<-ifelse(totpred_s150[,3]==totpred_s150[,7],2,0)
PredStars3_s150<-ifelse(totpred_s150[,4]==totpred_s150[,7],3,0)
PredStars4_s150<-ifelse(totpred_s150[,5]==totpred_s150[,7],4,0)
PredStars5_s150<-ifelse(totpred_s150[,6]==totpred_s150[,7],5,0)
totpred_s150$PredStars<-PredStars1_s150+PredStars2_s150+PredStars3_s150+PredStars4_s150+PredStars5_s150

# confusion matrix on test set
CM_pred_s150<-table(totpred_s150$PredStars,totpred_s150$TrainStars)
acc_pred_s150<-sum(ifelse(totpred_s150[,1]==totpred_s150[,8],1,0))/nrow(totpred_s150)
saveRDS(CM_pred_s150, "CM_pred_s150.RDS")
saveRDS(acc_pred_s150, "acc_pred_s150.RDS")

# predict on validation set
val1_s150<-as.matrix(predict(model1_s150,newdata=as.data.frame(testing_s150), type="response"))
val2_s150<-as.matrix(predict(model2_s150,newdata=as.data.frame(testing_s150), type="response"))
val3_s150<-as.matrix(predict(model3_s150,newdata=as.data.frame(testing_s150), type="response"))
val4_s150<-as.matrix(predict(model4_s150,newdata=as.data.frame(testing_s150), type="response"))
val5_s150<-as.matrix(predict(model5_s150,newdata=as.data.frame(testing_s150), type="response"))

# classify on validation set
valpred_s150<-as.data.frame(cbind(RealStars,val1_s150,val2_s150,val3_s150,val4_s150,val5_s150))
valpred_s150[,"max"]<-apply(valpred_s150[,2:6],1,max)
ValStars1_s150<-ifelse(valpred_s150[,2]==valpred_s150[,7],1,0)
ValStars2_s150<-ifelse(valpred_s150[,3]==valpred_s150[,7],2,0)
ValStars3_s150<-ifelse(valpred_s150[,4]==valpred_s150[,7],3,0)
ValStars4_s150<-ifelse(valpred_s150[,5]==valpred_s150[,7],4,0)
ValStars5_s150<-ifelse(valpred_s150[,6]==valpred_s150[,7],5,0)
valpred_s150$ValStars<-ValStars1_s150+ValStars2_s150+ValStars3_s150+ValStars4_s150+ValStars5_s150
CM_val_s150<-table(valpred_s150$ValStars,valpred_s150$RealStars)
acc_val_s150<-sum(ifelse(valpred_s150[,1]==valpred_s150[,8],1,0))/nrow(valpred_s150)
saveRDS(CM_val_s150, "CM_val_s150.RDS")
saveRDS(acc_val_s150, "acc_val_s150.RDS")


# simple 250 features

train1_s250<-as.data.frame(cbind(training_s250,y1))
model1_s250 <- glm(y1 ~ .,family=binomial(),data = train1_s250, control = list(maxit = 100))
pred1_s250 <- as.matrix(fitted(model1_s250))

train2_s250<-as.data.frame(cbind(training_s250,y2))
model2_s250 <- glm(y2 ~ .,family=binomial(),data = train2_s250, control = list(maxit = 100))
pred2_s250 <- as.matrix(fitted(model2_s250))

train3_s250<-as.data.frame(cbind(training_s250,y3))
model3_s250 <- glm(y3 ~ .,family=binomial(),data = train3_s250, control = list(maxit = 100))
pred3_s250 <- as.matrix(fitted(model3_s250))

train4_s250<-as.data.frame(cbind(training_s250,y4))
model4_s250 <- glm(y4 ~ .,family=binomial(),data = train4_s250, control = list(maxit = 100))
pred4_s250 <- as.matrix(fitted(model4_s250))

train5_s250<-as.data.frame(cbind(training_s250,y5))
model5_s250 <- glm(y5 ~ .,family=binomial(),data = train5_s250, control = list(maxit = 100))
pred5_s250 <- as.matrix(fitted(model5_s250))

# determine prediction on test set
totpred_s250<-as.data.frame(cbind(TrainStars,pred1_s250,pred2_s250,pred3_s250,pred4_s250,pred5_s250))
totpred_s250[,"max"]<-apply(totpred_s250[,2:6],1,max)
PredStars1_s250<-ifelse(totpred_s250[,2]==totpred_s250[,7],1,0)
PredStars2_s250<-ifelse(totpred_s250[,3]==totpred_s250[,7],2,0)
PredStars3_s250<-ifelse(totpred_s250[,4]==totpred_s250[,7],3,0)
PredStars4_s250<-ifelse(totpred_s250[,5]==totpred_s250[,7],4,0)
PredStars5_s250<-ifelse(totpred_s250[,6]==totpred_s250[,7],5,0)
totpred_s250$PredStars<-PredStars1_s250+PredStars2_s250+PredStars3_s250+PredStars4_s250+PredStars5_s250

# confusion matrix on test set
CM_pred_s250<-table(totpred_s250$PredStars,totpred_s250$TrainStars)
acc_pred_s250<-sum(ifelse(totpred_s250[,1]==totpred_s250[,8],1,0))/nrow(totpred_s250)
saveRDS(CM_pred_s250, "CM_pred_s250.RDS")
saveRDS(acc_pred_s250, "acc_pred_s250.RDS")

# predict on validation set
val1_s250<-as.matrix(predict(model1_s250,newdata=as.data.frame(testing_s250), type="response"))
val2_s250<-as.matrix(predict(model2_s250,newdata=as.data.frame(testing_s250), type="response"))
val3_s250<-as.matrix(predict(model3_s250,newdata=as.data.frame(testing_s250), type="response"))
val4_s250<-as.matrix(predict(model4_s250,newdata=as.data.frame(testing_s250), type="response"))
val5_s250<-as.matrix(predict(model5_s250,newdata=as.data.frame(testing_s250), type="response"))

# classify on validation set
valpred_s250<-as.data.frame(cbind(RealStars,val1_s250,val2_s250,val3_s250,val4_s250,val5_s250))
valpred_s250[,"max"]<-apply(valpred_s250[,2:6],1,max)
ValStars1_s250<-ifelse(valpred_s250[,2]==valpred_s250[,7],1,0)
ValStars2_s250<-ifelse(valpred_s250[,3]==valpred_s250[,7],2,0)
ValStars3_s250<-ifelse(valpred_s250[,4]==valpred_s250[,7],3,0)
ValStars4_s250<-ifelse(valpred_s250[,5]==valpred_s250[,7],4,0)
ValStars5_s250<-ifelse(valpred_s250[,6]==valpred_s250[,7],5,0)
valpred_s250$ValStars<-ValStars1_s250+ValStars2_s250+ValStars3_s250+ValStars4_s250+ValStars5_s250
CM_val_s250<-table(valpred_s250$ValStars,valpred_s250$RealStars)
acc_val_s250<-sum(ifelse(valpred_s250[,1]==valpred_s250[,8],1,0))/nrow(valpred_s250)
saveRDS(CM_val_s250, "CM_val_s250.RDS")
saveRDS(acc_val_s250, "acc_val_s250.RDS")



# simple 500 features

train1_s500<-as.data.frame(cbind(training_s500,y1))
model1_s500 <- glm(y1 ~ .,family=binomial(),data = train1_s500, control = list(maxit = 100))
pred1_s500 <- as.matrix(fitted(model1_s500))

train2_s500<-as.data.frame(cbind(training_s500,y2))
model2_s500 <- glm(y2 ~ .,family=binomial(),data = train2_s500, control = list(maxit = 100))
pred2_s500 <- as.matrix(fitted(model2_s500))

train3_s500<-as.data.frame(cbind(training_s500,y3))
model3_s500 <- glm(y3 ~ .,family=binomial(),data = train3_s500, control = list(maxit = 100))
pred3_s500 <- as.matrix(fitted(model3_s500))

train4_s500<-as.data.frame(cbind(training_s500,y4))
model4_s500 <- glm(y4 ~ .,family=binomial(),data = train4_s500, control = list(maxit = 100))
pred4_s500 <- as.matrix(fitted(model4_s500))

train5_s500<-as.data.frame(cbind(training_s500,y5))
model5_s500 <- glm(y5 ~ .,family=binomial(),data = train5_s500, control = list(maxit = 100))
pred5_s500 <- as.matrix(fitted(model5_s500))

# determine prediction on test set
totpred_s500<-as.data.frame(cbind(TrainStars,pred1_s500,pred2_s500,pred3_s500,pred4_s500,pred5_s500))
totpred_s500[,"max"]<-apply(totpred_s500[,2:6],1,max)
PredStars1_s500<-ifelse(totpred_s500[,2]==totpred_s500[,7],1,0)
PredStars2_s500<-ifelse(totpred_s500[,3]==totpred_s500[,7],2,0)
PredStars3_s500<-ifelse(totpred_s500[,4]==totpred_s500[,7],3,0)
PredStars4_s500<-ifelse(totpred_s500[,5]==totpred_s500[,7],4,0)
PredStars5_s500<-ifelse(totpred_s500[,6]==totpred_s500[,7],5,0)
totpred_s500$PredStars<-PredStars1_s500+PredStars2_s500+PredStars3_s500+PredStars4_s500+PredStars5_s500

# confusion matrix on test set
CM_pred_s500<-table(totpred_s500$PredStars,totpred_s500$TrainStars)
acc_pred_s500<-sum(ifelse(totpred_s500[,1]==totpred_s500[,8],1,0))/nrow(totpred_s500)
saveRDS(CM_pred_s500, "CM_pred_s500.RDS")
saveRDS(acc_pred_s500, "acc_pred_s500.RDS")

# predict on validation set
val1_s500<-as.matrix(predict(model1_s500,newdata=as.data.frame(testing_s500), type="response"))
val2_s500<-as.matrix(predict(model2_s500,newdata=as.data.frame(testing_s500), type="response"))
val3_s500<-as.matrix(predict(model3_s500,newdata=as.data.frame(testing_s500), type="response"))
val4_s500<-as.matrix(predict(model4_s500,newdata=as.data.frame(testing_s500), type="response"))
val5_s500<-as.matrix(predict(model5_s500,newdata=as.data.frame(testing_s500), type="response"))

# classify on validation set
valpred_s500<-as.data.frame(cbind(RealStars,val1_s500,val2_s500,val3_s500,val4_s500,val5_s500))
valpred_s500[,"max"]<-apply(valpred_s500[,2:6],1,max)
ValStars1_s500<-ifelse(valpred_s500[,2]==valpred_s500[,7],1,0)
ValStars2_s500<-ifelse(valpred_s500[,3]==valpred_s500[,7],2,0)
ValStars3_s500<-ifelse(valpred_s500[,4]==valpred_s500[,7],3,0)
ValStars4_s500<-ifelse(valpred_s500[,5]==valpred_s500[,7],4,0)
ValStars5_s500<-ifelse(valpred_s500[,6]==valpred_s500[,7],5,0)
valpred_s500$ValStars<-ValStars1_s500+ValStars2_s500+ValStars3_s500+ValStars4_s500+ValStars5_s500
CM_val_s500<-table(valpred_s500$ValStars,valpred_s500$RealStars)
acc_val_s500<-sum(ifelse(valpred_s500[,1]==valpred_s500[,8],1,0))/nrow(valpred_s500)
saveRDS(CM_val_s500, "CM_val_s500.RDS")
saveRDS(acc_val_s500, "acc_val_s500.RDS")




# bigrams 100 features

train1_b100<-as.data.frame(cbind(training_b100,y1))
model1_b100 <- glm(y1 ~ .,family=binomial(),data = train1_b100, control = list(maxit = 100))
pred1_b100 <- as.matrix(fitted(model1_b100))

train2_b100<-as.data.frame(cbind(training_b100,y2))
model2_b100 <- glm(y2 ~ .,family=binomial(),data = train2_b100, control = list(maxit = 100))
pred2_b100 <- as.matrix(fitted(model2_b100))

train3_b100<-as.data.frame(cbind(training_b100,y3))
model3_b100 <- glm(y3 ~ .,family=binomial(),data = train3_b100, control = list(maxit = 100))
pred3_b100 <- as.matrix(fitted(model3_b100))

train4_b100<-as.data.frame(cbind(training_b100,y4))
model4_b100 <- glm(y4 ~ .,family=binomial(),data = train4_b100, control = list(maxit = 100))
pred4_b100 <- as.matrix(fitted(model4_b100))

train5_b100<-as.data.frame(cbind(training_b100,y5))
model5_b100 <- glm(y5 ~ .,family=binomial(),data = train5_b100, control = list(maxit = 100))
pred5_b100 <- as.matrix(fitted(model5_b100))

# determine prediction on test set
totpred_b100<-as.data.frame(cbind(TrainStars,pred1_b100,pred2_b100,pred3_b100,pred4_b100,pred5_b100))
totpred_b100[,"max"]<-apply(totpred_b100[,2:6],1,max)
PredStars1_b100<-ifelse(totpred_b100[,2]==totpred_b100[,7],1,0)
PredStars2_b100<-ifelse(totpred_b100[,3]==totpred_b100[,7],2,0)
PredStars3_b100<-ifelse(totpred_b100[,4]==totpred_b100[,7],3,0)
PredStars4_b100<-ifelse(totpred_b100[,5]==totpred_b100[,7],4,0)
PredStars5_b100<-ifelse(totpred_b100[,6]==totpred_b100[,7],5,0)
totpred_b100$PredStars<-PredStars1_b100+PredStars2_b100+PredStars3_b100+PredStars4_b100+PredStars5_b100

# confusion matrix on test set
CM_pred_b100<-table(totpred_b100$PredStars,totpred_b100$TrainStars)
acc_pred_b100<-sum(ifelse(totpred_b100[,1]==totpred_b100[,8],1,0))/nrow(totpred_b100)
saveRDS(CM_pred_b100, "CM_pred_b100.RDS")
saveRDS(acc_pred_b100, "acc_pred_b100.RDS")

# predict on validation set
val1_b100<-as.matrix(predict(model1_b100,newdata=as.data.frame(testing_b100), type="response"))
val2_b100<-as.matrix(predict(model2_b100,newdata=as.data.frame(testing_b100), type="response"))
val3_b100<-as.matrix(predict(model3_b100,newdata=as.data.frame(testing_b100), type="response"))
val4_b100<-as.matrix(predict(model4_b100,newdata=as.data.frame(testing_b100), type="response"))
val5_b100<-as.matrix(predict(model5_b100,newdata=as.data.frame(testing_b100), type="response"))

# classify on validation set
valpred_b100<-as.data.frame(cbind(RealStars,val1_b100,val2_b100,val3_b100,val4_b100,val5_b100))
valpred_b100[,"max"]<-apply(valpred_b100[,2:6],1,max)
ValStars1_b100<-ifelse(valpred_b100[,2]==valpred_b100[,7],1,0)
ValStars2_b100<-ifelse(valpred_b100[,3]==valpred_b100[,7],2,0)
ValStars3_b100<-ifelse(valpred_b100[,4]==valpred_b100[,7],3,0)
ValStars4_b100<-ifelse(valpred_b100[,5]==valpred_b100[,7],4,0)
ValStars5_b100<-ifelse(valpred_b100[,6]==valpred_b100[,7],5,0)
valpred_b100$ValStars<-ValStars1_b100+ValStars2_b100+ValStars3_b100+ValStars4_b100+ValStars5_b100
CM_val_b100<-table(valpred_b100$ValStars,valpred_b100$RealStars)
acc_val_b100<-sum(ifelse(valpred_b100[,1]==valpred_b100[,8],1,0))/nrow(valpred_b100)
saveRDS(CM_val_b100, "CM_val_b100.RDS")
saveRDS(acc_val_b100, "acc_val_b100.RDS")


# bigrams 150 features

train1_b150<-as.data.frame(cbind(training_b150,y1))
model1_b150 <- glm(y1 ~ .,family=binomial(),data = train1_b150, control = list(maxit = 100))
pred1_b150 <- as.matrix(fitted(model1_b150))

train2_b150<-as.data.frame(cbind(training_b150,y2))
model2_b150 <- glm(y2 ~ .,family=binomial(),data = train2_b150, control = list(maxit = 100))
pred2_b150 <- as.matrix(fitted(model2_b150))

train3_b150<-as.data.frame(cbind(training_b150,y3))
model3_b150 <- glm(y3 ~ .,family=binomial(),data = train3_b150, control = list(maxit = 100))
pred3_b150 <- as.matrix(fitted(model3_b150))

train4_b150<-as.data.frame(cbind(training_b150,y4))
model4_b150 <- glm(y4 ~ .,family=binomial(),data = train4_b150, control = list(maxit = 100))
pred4_b150 <- as.matrix(fitted(model4_b150))

train5_b150<-as.data.frame(cbind(training_b150,y5))
model5_b150 <- glm(y5 ~ .,family=binomial(),data = train5_b150, control = list(maxit = 100))
pred5_b150 <- as.matrix(fitted(model5_b150))

# determine prediction on test set
totpred_b150<-as.data.frame(cbind(TrainStars,pred1_b150,pred2_b150,pred3_b150,pred4_b150,pred5_b150))
totpred_b150[,"max"]<-apply(totpred_b150[,2:6],1,max)
PredStars1_b150<-ifelse(totpred_b150[,2]==totpred_b150[,7],1,0)
PredStars2_b150<-ifelse(totpred_b150[,3]==totpred_b150[,7],2,0)
PredStars3_b150<-ifelse(totpred_b150[,4]==totpred_b150[,7],3,0)
PredStars4_b150<-ifelse(totpred_b150[,5]==totpred_b150[,7],4,0)
PredStars5_b150<-ifelse(totpred_b150[,6]==totpred_b150[,7],5,0)
totpred_b150$PredStars<-PredStars1_b150+PredStars2_b150+PredStars3_b150+PredStars4_b150+PredStars5_b150

# confusion matrix on test set
CM_pred_b150<-table(totpred_b150$PredStars,totpred_b150$TrainStars)
acc_pred_b150<-sum(ifelse(totpred_b150[,1]==totpred_b150[,8],1,0))/nrow(totpred_b150)
saveRDS(CM_pred_b150, "CM_pred_b150.RDS")
saveRDS(acc_pred_b150, "acc_pred_b150.RDS")

# predict on validation set
val1_b150<-as.matrix(predict(model1_b150,newdata=as.data.frame(testing_b150), type="response"))
val2_b150<-as.matrix(predict(model2_b150,newdata=as.data.frame(testing_b150), type="response"))
val3_b150<-as.matrix(predict(model3_b150,newdata=as.data.frame(testing_b150), type="response"))
val4_b150<-as.matrix(predict(model4_b150,newdata=as.data.frame(testing_b150), type="response"))
val5_b150<-as.matrix(predict(model5_b150,newdata=as.data.frame(testing_b150), type="response"))

# classify on validation set
valpred_b150<-as.data.frame(cbind(RealStars,val1_b150,val2_b150,val3_b150,val4_b150,val5_b150))
valpred_b150[,"max"]<-apply(valpred_b150[,2:6],1,max)
ValStars1_b150<-ifelse(valpred_b150[,2]==valpred_b150[,7],1,0)
ValStars2_b150<-ifelse(valpred_b150[,3]==valpred_b150[,7],2,0)
ValStars3_b150<-ifelse(valpred_b150[,4]==valpred_b150[,7],3,0)
ValStars4_b150<-ifelse(valpred_b150[,5]==valpred_b150[,7],4,0)
ValStars5_b150<-ifelse(valpred_b150[,6]==valpred_b150[,7],5,0)
valpred_b150$ValStars<-ValStars1_b150+ValStars2_b150+ValStars3_b150+ValStars4_b150+ValStars5_b150
CM_val_b150<-table(valpred_b150$ValStars,valpred_b150$RealStars)
acc_val_b150<-sum(ifelse(valpred_b150[,1]==valpred_b150[,8],1,0))/nrow(valpred_b150)
saveRDS(CM_val_b150, "CM_val_b150.RDS")
saveRDS(acc_val_b150, "acc_val_b150.RDS")


# bigrams 250 features

train1_b250<-as.data.frame(cbind(training_b250,y1))
model1_b250 <- glm(y1 ~ .,family=binomial(),data = train1_b250, control = list(maxit = 100))
pred1_b250 <- as.matrix(fitted(model1_b250))

train2_b250<-as.data.frame(cbind(training_b250,y2))
model2_b250 <- glm(y2 ~ .,family=binomial(),data = train2_b250, control = list(maxit = 100))
pred2_b250 <- as.matrix(fitted(model2_b250))

train3_b250<-as.data.frame(cbind(training_b250,y3))
model3_b250 <- glm(y3 ~ .,family=binomial(),data = train3_b250, control = list(maxit = 100))
pred3_b250 <- as.matrix(fitted(model3_b250))

train4_b250<-as.data.frame(cbind(training_b250,y4))
model4_b250 <- glm(y4 ~ .,family=binomial(),data = train4_b250, control = list(maxit = 100))
pred4_b250 <- as.matrix(fitted(model4_b250))

train5_b250<-as.data.frame(cbind(training_b250,y5))
model5_b250 <- glm(y5 ~ .,family=binomial(),data = train5_b250, control = list(maxit = 100))
pred5_b250 <- as.matrix(fitted(model5_b250))

# determine prediction on test set
totpred_b250<-as.data.frame(cbind(TrainStars,pred1_b250,pred2_b250,pred3_b250,pred4_b250,pred5_b250))
totpred_b250[,"max"]<-apply(totpred_b250[,2:6],1,max)
PredStars1_b250<-ifelse(totpred_b250[,2]==totpred_b250[,7],1,0)
PredStars2_b250<-ifelse(totpred_b250[,3]==totpred_b250[,7],2,0)
PredStars3_b250<-ifelse(totpred_b250[,4]==totpred_b250[,7],3,0)
PredStars4_b250<-ifelse(totpred_b250[,5]==totpred_b250[,7],4,0)
PredStars5_b250<-ifelse(totpred_b250[,6]==totpred_b250[,7],5,0)
totpred_b250$PredStars<-PredStars1_b250+PredStars2_b250+PredStars3_b250+PredStars4_b250+PredStars5_b250

# confusion matrix on test set
CM_pred_b250<-table(totpred_b250$PredStars,totpred_b250$TrainStars)
acc_pred_b250<-sum(ifelse(totpred_b250[,1]==totpred_b250[,8],1,0))/nrow(totpred_b250)
saveRDS(CM_pred_b250, "CM_pred_b250.RDS")
saveRDS(acc_pred_b250, "acc_pred_b250.RDS")

# predict on validation set
val1_b250<-as.matrix(predict(model1_b250,newdata=as.data.frame(testing_b250), type="response"))
val2_b250<-as.matrix(predict(model2_b250,newdata=as.data.frame(testing_b250), type="response"))
val3_b250<-as.matrix(predict(model3_b250,newdata=as.data.frame(testing_b250), type="response"))
val4_b250<-as.matrix(predict(model4_b250,newdata=as.data.frame(testing_b250), type="response"))
val5_b250<-as.matrix(predict(model5_b250,newdata=as.data.frame(testing_b250), type="response"))

# classify on validation set
valpred_b250<-as.data.frame(cbind(RealStars,val1_b250,val2_b250,val3_b250,val4_b250,val5_b250))
valpred_b250[,"max"]<-apply(valpred_b250[,2:6],1,max)
ValStars1_b250<-ifelse(valpred_b250[,2]==valpred_b250[,7],1,0)
ValStars2_b250<-ifelse(valpred_b250[,3]==valpred_b250[,7],2,0)
ValStars3_b250<-ifelse(valpred_b250[,4]==valpred_b250[,7],3,0)
ValStars4_b250<-ifelse(valpred_b250[,5]==valpred_b250[,7],4,0)
ValStars5_b250<-ifelse(valpred_b250[,6]==valpred_b250[,7],5,0)
valpred_b250$ValStars<-ValStars1_b250+ValStars2_b250+ValStars3_b250+ValStars4_b250+ValStars5_b250
CM_val_b250<-table(valpred_b250$ValStars,valpred_b250$RealStars)
acc_val_b250<-sum(ifelse(valpred_b250[,1]==valpred_b250[,8],1,0))/nrow(valpred_b250)
saveRDS(CM_val_b250, "CM_val_b250.RDS")
saveRDS(acc_val_b250, "acc_val_b250.RDS")



# bigrams 500 features

train1_b500<-as.data.frame(cbind(training_b500,y1))
model1_b500 <- glm(y1 ~ .,family=binomial(),data = train1_b500, control = list(maxit = 100))
pred1_b500 <- as.matrix(fitted(model1_b500))

train2_b500<-as.data.frame(cbind(training_b500,y2))
model2_b500 <- glm(y2 ~ .,family=binomial(),data = train2_b500, control = list(maxit = 100))
pred2_b500 <- as.matrix(fitted(model2_b500))

train3_b500<-as.data.frame(cbind(training_b500,y3))
model3_b500 <- glm(y3 ~ .,family=binomial(),data = train3_b500, control = list(maxit = 100))
pred3_b500 <- as.matrix(fitted(model3_b500))

train4_b500<-as.data.frame(cbind(training_b500,y4))
model4_b500 <- glm(y4 ~ .,family=binomial(),data = train4_b500, control = list(maxit = 100))
pred4_b500 <- as.matrix(fitted(model4_b500))

train5_b500<-as.data.frame(cbind(training_b500,y5))
model5_b500 <- glm(y5 ~ .,family=binomial(),data = train5_b500, control = list(maxit = 100))
pred5_b500 <- as.matrix(fitted(model5_b500))

# determine prediction on test set
totpred_b500<-as.data.frame(cbind(TrainStars,pred1_b500,pred2_b500,pred3_b500,pred4_b500,pred5_b500))
totpred_b500[,"max"]<-apply(totpred_b500[,2:6],1,max)
PredStars1_b500<-ifelse(totpred_b500[,2]==totpred_b500[,7],1,0)
PredStars2_b500<-ifelse(totpred_b500[,3]==totpred_b500[,7],2,0)
PredStars3_b500<-ifelse(totpred_b500[,4]==totpred_b500[,7],3,0)
PredStars4_b500<-ifelse(totpred_b500[,5]==totpred_b500[,7],4,0)
PredStars5_b500<-ifelse(totpred_b500[,6]==totpred_b500[,7],5,0)
totpred_b500$PredStars<-PredStars1_b500+PredStars2_b500+PredStars3_b500+PredStars4_b500+PredStars5_b500

# confusion matrix on test set
CM_pred_b500<-table(totpred_b500$PredStars,totpred_b500$TrainStars)
acc_pred_b500<-sum(ifelse(totpred_b500[,1]==totpred_b500[,8],1,0))/nrow(totpred_b500)
saveRDS(CM_pred_b500, "CM_pred_b500.RDS")
saveRDS(acc_pred_b500, "acc_pred_b500.RDS")

# predict on validation set
val1_b500<-as.matrix(predict(model1_b500,newdata=as.data.frame(testing_b500), type="response"))
val2_b500<-as.matrix(predict(model2_b500,newdata=as.data.frame(testing_b500), type="response"))
val3_b500<-as.matrix(predict(model3_b500,newdata=as.data.frame(testing_b500), type="response"))
val4_b500<-as.matrix(predict(model4_b500,newdata=as.data.frame(testing_b500), type="response"))
val5_b500<-as.matrix(predict(model5_b500,newdata=as.data.frame(testing_b500), type="response"))

# classify on validation set
valpred_b500<-as.data.frame(cbind(RealStars,val1_b500,val2_b500,val3_b500,val4_b500,val5_b500))
valpred_b500[,"max"]<-apply(valpred_b500[,2:6],1,max)
ValStars1_b500<-ifelse(valpred_b500[,2]==valpred_b500[,7],1,0)
ValStars2_b500<-ifelse(valpred_b500[,3]==valpred_b500[,7],2,0)
ValStars3_b500<-ifelse(valpred_b500[,4]==valpred_b500[,7],3,0)
ValStars4_b500<-ifelse(valpred_b500[,5]==valpred_b500[,7],4,0)
ValStars5_b500<-ifelse(valpred_b500[,6]==valpred_b500[,7],5,0)
valpred_b500$ValStars<-ValStars1_b500+ValStars2_b500+ValStars3_b500+ValStars4_b500+ValStars5_b500
CM_val_b500<-table(valpred_b500$ValStars,valpred_b500$RealStars)
acc_val_b500<-sum(ifelse(valpred_b500[,1]==valpred_b500[,8],1,0))/nrow(valpred_b500)
saveRDS(CM_val_b500, "CM_val_b500.RDS")
saveRDS(acc_val_b500, "acc_val_b500.RDS")




# adjectives 100 features

train1_a100<-as.data.frame(cbind(training_a100,y1))
model1_a100 <- glm(y1 ~ .,family=binomial(),data = train1_a100, control = list(maxit = 100))
pred1_a100 <- as.matrix(fitted(model1_a100))

train2_a100<-as.data.frame(cbind(training_a100,y2))
model2_a100 <- glm(y2 ~ .,family=binomial(),data = train2_a100, control = list(maxit = 100))
pred2_a100 <- as.matrix(fitted(model2_a100))

train3_a100<-as.data.frame(cbind(training_a100,y3))
model3_a100 <- glm(y3 ~ .,family=binomial(),data = train3_a100, control = list(maxit = 100))
pred3_a100 <- as.matrix(fitted(model3_a100))

train4_a100<-as.data.frame(cbind(training_a100,y4))
model4_a100 <- glm(y4 ~ .,family=binomial(),data = train4_a100, control = list(maxit = 100))
pred4_a100 <- as.matrix(fitted(model4_a100))

train5_a100<-as.data.frame(cbind(training_a100,y5))
model5_a100 <- glm(y5 ~ .,family=binomial(),data = train5_a100, control = list(maxit = 100))
pred5_a100 <- as.matrix(fitted(model5_a100))

# determine prediction on test set
totpred_a100<-as.data.frame(cbind(TrainStars,pred1_a100,pred2_a100,pred3_a100,pred4_a100,pred5_a100))
totpred_a100[,"max"]<-apply(totpred_a100[,2:6],1,max)
PredStars1_a100<-ifelse(totpred_a100[,2]==totpred_a100[,7],1,0)
PredStars2_a100<-ifelse(totpred_a100[,3]==totpred_a100[,7],2,0)
PredStars3_a100<-ifelse(totpred_a100[,4]==totpred_a100[,7],3,0)
PredStars4_a100<-ifelse(totpred_a100[,5]==totpred_a100[,7],4,0)
PredStars5_a100<-ifelse(totpred_a100[,6]==totpred_a100[,7],5,0)
totpred_a100$PredStars<-PredStars1_a100+PredStars2_a100+PredStars3_a100+PredStars4_a100+PredStars5_a100

# confusion matrix on test set
CM_pred_a100<-table(totpred_a100$PredStars,totpred_a100$TrainStars)
acc_pred_a100<-sum(ifelse(totpred_a100[,1]==totpred_a100[,8],1,0))/nrow(totpred_a100)
saveRDS(CM_pred_a100, "CM_pred_a100.RDS")
saveRDS(acc_pred_a100, "acc_pred_a100.RDS")

# predict on validation set
val1_a100<-as.matrix(predict(model1_a100,newdata=as.data.frame(testing_a100), type="response"))
val2_a100<-as.matrix(predict(model2_a100,newdata=as.data.frame(testing_a100), type="response"))
val3_a100<-as.matrix(predict(model3_a100,newdata=as.data.frame(testing_a100), type="response"))
val4_a100<-as.matrix(predict(model4_a100,newdata=as.data.frame(testing_a100), type="response"))
val5_a100<-as.matrix(predict(model5_a100,newdata=as.data.frame(testing_a100), type="response"))

# classify on validation set
valpred_a100<-as.data.frame(cbind(RealStars,val1_a100,val2_a100,val3_a100,val4_a100,val5_a100))
valpred_a100[,"max"]<-apply(valpred_a100[,2:6],1,max)
ValStars1_a100<-ifelse(valpred_a100[,2]==valpred_a100[,7],1,0)
ValStars2_a100<-ifelse(valpred_a100[,3]==valpred_a100[,7],2,0)
ValStars3_a100<-ifelse(valpred_a100[,4]==valpred_a100[,7],3,0)
ValStars4_a100<-ifelse(valpred_a100[,5]==valpred_a100[,7],4,0)
ValStars5_a100<-ifelse(valpred_a100[,6]==valpred_a100[,7],5,0)
valpred_a100$ValStars<-ValStars1_a100+ValStars2_a100+ValStars3_a100+ValStars4_a100+ValStars5_a100
CM_val_a100<-table(valpred_a100$ValStars,valpred_a100$RealStars)
acc_val_a100<-sum(ifelse(valpred_a100[,1]==valpred_a100[,8],1,0))/nrow(valpred_a100)
saveRDS(CM_val_a100, "CM_val_a100.RDS")
saveRDS(acc_val_a100, "acc_val_a100.RDS")


# adjectives 150 features

train1_a150<-as.data.frame(cbind(training_a150,y1))
model1_a150 <- glm(y1 ~ .,family=binomial(),data = train1_a150, control = list(maxit = 100))
pred1_a150 <- as.matrix(fitted(model1_a150))

train2_a150<-as.data.frame(cbind(training_a150,y2))
model2_a150 <- glm(y2 ~ .,family=binomial(),data = train2_a150, control = list(maxit = 100))
pred2_a150 <- as.matrix(fitted(model2_a150))

train3_a150<-as.data.frame(cbind(training_a150,y3))
model3_a150 <- glm(y3 ~ .,family=binomial(),data = train3_a150, control = list(maxit = 100))
pred3_a150 <- as.matrix(fitted(model3_a150))

train4_a150<-as.data.frame(cbind(training_a150,y4))
model4_a150 <- glm(y4 ~ .,family=binomial(),data = train4_a150, control = list(maxit = 100))
pred4_a150 <- as.matrix(fitted(model4_a150))

train5_a150<-as.data.frame(cbind(training_a150,y5))
model5_a150 <- glm(y5 ~ .,family=binomial(),data = train5_a150, control = list(maxit = 100))
pred5_a150 <- as.matrix(fitted(model5_a150))

# determine prediction on test set
totpred_a150<-as.data.frame(cbind(TrainStars,pred1_a150,pred2_a150,pred3_a150,pred4_a150,pred5_a150))
totpred_a150[,"max"]<-apply(totpred_a150[,2:6],1,max)
PredStars1_a150<-ifelse(totpred_a150[,2]==totpred_a150[,7],1,0)
PredStars2_a150<-ifelse(totpred_a150[,3]==totpred_a150[,7],2,0)
PredStars3_a150<-ifelse(totpred_a150[,4]==totpred_a150[,7],3,0)
PredStars4_a150<-ifelse(totpred_a150[,5]==totpred_a150[,7],4,0)
PredStars5_a150<-ifelse(totpred_a150[,6]==totpred_a150[,7],5,0)
totpred_a150$PredStars<-PredStars1_a150+PredStars2_a150+PredStars3_a150+PredStars4_a150+PredStars5_a150

# confusion matrix on test set
CM_pred_a150<-table(totpred_a150$PredStars,totpred_a150$TrainStars)
acc_pred_a150<-sum(ifelse(totpred_a150[,1]==totpred_a150[,8],1,0))/nrow(totpred_a150)
saveRDS(CM_pred_a150, "CM_pred_a150.RDS")
saveRDS(acc_pred_a150, "acc_pred_a150.RDS")

# predict on validation set
val1_a150<-as.matrix(predict(model1_a150,newdata=as.data.frame(testing_a150), type="response"))
val2_a150<-as.matrix(predict(model2_a150,newdata=as.data.frame(testing_a150), type="response"))
val3_a150<-as.matrix(predict(model3_a150,newdata=as.data.frame(testing_a150), type="response"))
val4_a150<-as.matrix(predict(model4_a150,newdata=as.data.frame(testing_a150), type="response"))
val5_a150<-as.matrix(predict(model5_a150,newdata=as.data.frame(testing_a150), type="response"))

# classify on validation set
valpred_a150<-as.data.frame(cbind(RealStars,val1_a150,val2_a150,val3_a150,val4_a150,val5_a150))
valpred_a150[,"max"]<-apply(valpred_a150[,2:6],1,max)
ValStars1_a150<-ifelse(valpred_a150[,2]==valpred_a150[,7],1,0)
ValStars2_a150<-ifelse(valpred_a150[,3]==valpred_a150[,7],2,0)
ValStars3_a150<-ifelse(valpred_a150[,4]==valpred_a150[,7],3,0)
ValStars4_a150<-ifelse(valpred_a150[,5]==valpred_a150[,7],4,0)
ValStars5_a150<-ifelse(valpred_a150[,6]==valpred_a150[,7],5,0)
valpred_a150$ValStars<-ValStars1_a150+ValStars2_a150+ValStars3_a150+ValStars4_a150+ValStars5_a150
CM_val_a150<-table(valpred_a150$ValStars,valpred_a150$RealStars)
acc_val_a150<-sum(ifelse(valpred_a150[,1]==valpred_a150[,8],1,0))/nrow(valpred_a150)
saveRDS(CM_val_a150, "CM_val_a150.RDS")
saveRDS(acc_val_a150, "acc_val_a150.RDS")


# adjectives 250 features

train1_a250<-as.data.frame(cbind(training_a250,y1))
model1_a250 <- glm(y1 ~ .,family=binomial(),data = train1_a250, control = list(maxit = 100))
pred1_a250 <- as.matrix(fitted(model1_a250))

train2_a250<-as.data.frame(cbind(training_a250,y2))
model2_a250 <- glm(y2 ~ .,family=binomial(),data = train2_a250, control = list(maxit = 100))
pred2_a250 <- as.matrix(fitted(model2_a250))

train3_a250<-as.data.frame(cbind(training_a250,y3))
model3_a250 <- glm(y3 ~ .,family=binomial(),data = train3_a250, control = list(maxit = 100))
pred3_a250 <- as.matrix(fitted(model3_a250))

train4_a250<-as.data.frame(cbind(training_a250,y4))
model4_a250 <- glm(y4 ~ .,family=binomial(),data = train4_a250, control = list(maxit = 100))
pred4_a250 <- as.matrix(fitted(model4_a250))

train5_a250<-as.data.frame(cbind(training_a250,y5))
model5_a250 <- glm(y5 ~ .,family=binomial(),data = train5_a250, control = list(maxit = 100))
pred5_a250 <- as.matrix(fitted(model5_a250))

# determine prediction on test set
totpred_a250<-as.data.frame(cbind(TrainStars,pred1_a250,pred2_a250,pred3_a250,pred4_a250,pred5_a250))
totpred_a250[,"max"]<-apply(totpred_a250[,2:6],1,max)
PredStars1_a250<-ifelse(totpred_a250[,2]==totpred_a250[,7],1,0)
PredStars2_a250<-ifelse(totpred_a250[,3]==totpred_a250[,7],2,0)
PredStars3_a250<-ifelse(totpred_a250[,4]==totpred_a250[,7],3,0)
PredStars4_a250<-ifelse(totpred_a250[,5]==totpred_a250[,7],4,0)
PredStars5_a250<-ifelse(totpred_a250[,6]==totpred_a250[,7],5,0)
totpred_a250$PredStars<-PredStars1_a250+PredStars2_a250+PredStars3_a250+PredStars4_a250+PredStars5_a250

# confusion matrix on test set
CM_pred_a250<-table(totpred_a250$PredStars,totpred_a250$TrainStars)
acc_pred_a250<-sum(ifelse(totpred_a250[,1]==totpred_a250[,8],1,0))/nrow(totpred_a250)
saveRDS(CM_pred_a250, "CM_pred_a250.RDS")
saveRDS(acc_pred_a250, "acc_pred_a250.RDS")

# predict on validation set
val1_a250<-as.matrix(predict(model1_a250,newdata=as.data.frame(testing_a250), type="response"))
val2_a250<-as.matrix(predict(model2_a250,newdata=as.data.frame(testing_a250), type="response"))
val3_a250<-as.matrix(predict(model3_a250,newdata=as.data.frame(testing_a250), type="response"))
val4_a250<-as.matrix(predict(model4_a250,newdata=as.data.frame(testing_a250), type="response"))
val5_a250<-as.matrix(predict(model5_a250,newdata=as.data.frame(testing_a250), type="response"))

# classify on validation set
valpred_a250<-as.data.frame(cbind(RealStars,val1_a250,val2_a250,val3_a250,val4_a250,val5_a250))
valpred_a250[,"max"]<-apply(valpred_a250[,2:6],1,max)
ValStars1_a250<-ifelse(valpred_a250[,2]==valpred_a250[,7],1,0)
ValStars2_a250<-ifelse(valpred_a250[,3]==valpred_a250[,7],2,0)
ValStars3_a250<-ifelse(valpred_a250[,4]==valpred_a250[,7],3,0)
ValStars4_a250<-ifelse(valpred_a250[,5]==valpred_a250[,7],4,0)
ValStars5_a250<-ifelse(valpred_a250[,6]==valpred_a250[,7],5,0)
valpred_a250$ValStars<-ValStars1_a250+ValStars2_a250+ValStars3_a250+ValStars4_a250+ValStars5_a250
CM_val_a250<-table(valpred_a250$ValStars,valpred_a250$RealStars)
acc_val_a250<-sum(ifelse(valpred_a250[,1]==valpred_a250[,8],1,0))/nrow(valpred_a250)
saveRDS(CM_val_a250, "CM_val_a250.RDS")
saveRDS(acc_val_a250, "acc_val_a250.RDS")



# adjectives 500 features

train1_a500<-as.data.frame(cbind(training_a500,y1))
model1_a500 <- glm(y1 ~ .,family=binomial(),data = train1_a500, control = list(maxit = 100))
pred1_a500 <- as.matrix(fitted(model1_a500))

train2_a500<-as.data.frame(cbind(training_a500,y2))
model2_a500 <- glm(y2 ~ .,family=binomial(),data = train2_a500, control = list(maxit = 100))
pred2_a500 <- as.matrix(fitted(model2_a500))

train3_a500<-as.data.frame(cbind(training_a500,y3))
model3_a500 <- glm(y3 ~ .,family=binomial(),data = train3_a500, control = list(maxit = 100))
pred3_a500 <- as.matrix(fitted(model3_a500))

train4_a500<-as.data.frame(cbind(training_a500,y4))
model4_a500 <- glm(y4 ~ .,family=binomial(),data = train4_a500, control = list(maxit = 100))
pred4_a500 <- as.matrix(fitted(model4_a500))

train5_a500<-as.data.frame(cbind(training_a500,y5))
model5_a500 <- glm(y5 ~ .,family=binomial(),data = train5_a500, control = list(maxit = 100))
pred5_a500 <- as.matrix(fitted(model5_a500))

# determine prediction on test set
totpred_a500<-as.data.frame(cbind(TrainStars,pred1_a500,pred2_a500,pred3_a500,pred4_a500,pred5_a500))
totpred_a500[,"max"]<-apply(totpred_a500[,2:6],1,max)
PredStars1_a500<-ifelse(totpred_a500[,2]==totpred_a500[,7],1,0)
PredStars2_a500<-ifelse(totpred_a500[,3]==totpred_a500[,7],2,0)
PredStars3_a500<-ifelse(totpred_a500[,4]==totpred_a500[,7],3,0)
PredStars4_a500<-ifelse(totpred_a500[,5]==totpred_a500[,7],4,0)
PredStars5_a500<-ifelse(totpred_a500[,6]==totpred_a500[,7],5,0)
totpred_a500$PredStars<-PredStars1_a500+PredStars2_a500+PredStars3_a500+PredStars4_a500+PredStars5_a500

# confusion matrix on test set
CM_pred_a500<-table(totpred_a500$PredStars,totpred_a500$TrainStars)
acc_pred_a500<-sum(ifelse(totpred_a500[,1]==totpred_a500[,8],1,0))/nrow(totpred_a500)
saveRDS(CM_pred_a500, "CM_pred_a500.RDS")
saveRDS(acc_pred_a500, "acc_pred_a500.RDS")

# predict on validation set
val1_a500<-as.matrix(predict(model1_a500,newdata=as.data.frame(testing_a500), type="response"))
val2_a500<-as.matrix(predict(model2_a500,newdata=as.data.frame(testing_a500), type="response"))
val3_a500<-as.matrix(predict(model3_a500,newdata=as.data.frame(testing_a500), type="response"))
val4_a500<-as.matrix(predict(model4_a500,newdata=as.data.frame(testing_a500), type="response"))
val5_a500<-as.matrix(predict(model5_a500,newdata=as.data.frame(testing_a500), type="response"))

# classify on validation set
valpred_a500<-as.data.frame(cbind(RealStars,val1_a500,val2_a500,val3_a500,val4_a500,val5_a500))
valpred_a500[,"max"]<-apply(valpred_a500[,2:6],1,max)
ValStars1_a500<-ifelse(valpred_a500[,2]==valpred_a500[,7],1,0)
ValStars2_a500<-ifelse(valpred_a500[,3]==valpred_a500[,7],2,0)
ValStars3_a500<-ifelse(valpred_a500[,4]==valpred_a500[,7],3,0)
ValStars4_a500<-ifelse(valpred_a500[,5]==valpred_a500[,7],4,0)
ValStars5_a500<-ifelse(valpred_a500[,6]==valpred_a500[,7],5,0)
valpred_a500$ValStars<-ValStars1_a500+ValStars2_a500+ValStars3_a500+ValStars4_a500+ValStars5_a500
CM_val_a500<-table(valpred_a500$ValStars,valpred_a500$RealStars)
acc_val_a500<-sum(ifelse(valpred_a500[,1]==valpred_a500[,8],1,0))/nrow(valpred_a500)
saveRDS(CM_val_a500, "CM_val_a500.RDS")
saveRDS(acc_val_a500, "acc_val_a500.RDS")


# discrepant reviews
dis1<-valpred_s500$RealStars==1 & valpred_s500$ValStars ==5
dis5<-valpred_s500$RealStars==5 & valpred_s500$ValStars ==1

dtext1<-RealText[dis1]
dtext5<-RealText[dis5]

#show example 13 of dtext1
dtext1[13]
