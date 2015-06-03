library(data.table)
library(e1071)
library(tm)

fread('~/Downloads/training.csv')

data$Categorie1  = as.factor(data$Categorie1)
data$Categorie2  = as.factor(data$Categorie2)
data$Categorie3  = as.factor(data$Categorie3)
data$Produit_Cdiscount = as.factor(data$Produit_Cdiscount)

control=list(tolower=TRUE,removePunctuation=TRUE,removeNumbers=TRUE,stopwords=TRUE,
                                                 stemming=TRUE

dtm = DocumentTermMatrix(Corpus(VectorSource(data[ Produit_Cdiscount == 1 , Libelle ])),
											control=list(tolower=TRUE,removePunctuation=TRUE,removeNumbers=TRUE,stopwords=TRUE,
                                                 stemming=TRUE))

X = as.data.table(as.matrix(dtm))


 model = svm(sparse_train_train,train_train_relevance, kernel="linear", cost=0.3)

> length(unique(data$Categorie1))
#[1] 52
> length(unique(data$Categorie2))
#[1] 536
> length(unique(data$Categorie3))
#[1] 5789


