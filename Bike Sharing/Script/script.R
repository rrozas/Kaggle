library(data.table)
library(Metrics)
library(gbm)

train  <- fread("~/Documents/Kaggle/Bike Sharing/Data/train.csv")
test  <- fread("~/Documents/Kaggle/Bike Sharing/Data/test.csv")

l_factor = c( "season" , "workingday" , "holiday" , "weather")

for(c in l_factor){
  train[[c]]  <- as.factor( train[[c]] )
  test[[c]]  <- as.factor( test[[c]] )
}

train$date  <- as.Date(substr(train$datetime,1,10))
train$weekdays  <- weekdays(train$date)
train$year  <-  year(train$date)
train$hour  <- factor( as.integer( substr( train$datetime,11,13) ) , levels = 0:23, ordered = T)
train$hc  <- factor(train$hour %in% 2:5)
train$log_count  <- log1p(train$count)
train$windspeed[ train$windspeed == 0]  <-  NA

test$date  <- as.Date(substr(test$datetime,1,10))
test$weekdays  <- weekdays(test$date)
test$year  <-  year(test$date)
test$hour  <- factor( as.integer( substr( test$datetime,11,13) ) , levels = 0:23, ordered = T)
test$hc  <- factor(test$hour %in% 2:5)
test$windspeed[ test$windspeed == 0]  <-  NA

ndx  <- sample(1:nrow(train) , 0.6 * nrow(train))
tr  <- train[ndx]
te  <- train[-ndx]
formula <- log_count ~ season + holiday + workingday + weather + temp + atemp + humidity + hour + year + hc + windspeed
fit.gbm <- gbm(formula, data=tr, interaction.depth = 10, shrinkage = .03, train.fraction = .5, n.trees = 500, verbose = T )
p  <- predict(fit.gbm, te)
p <- expm1 (p)
rmsle(te$count, p)

i = c(5,7,10,15)
s = c(.003,.01,.03,.1)
n = c(500, 700 , 900)

param<- expand.grid(i,s,n)
res = c()
ev  <- function (p){
fit.gbm <- gbm(formula, data=tr, interaction.depth = p[1], shrinkage = p[2], train.fraction = 1, n.trees = p[3], verbose = F )
  pred  <- predict(fit.gbm, te, n.trees = p[3])
  pred <- expm1 (pred)
  return(rmsle(te$count, pred))
  
}

res <- apply(param, MARGIN = 1, ev)

fit.gbm <- gbm(formula, data=train, interaction.depth = 10, shrinkage = .03, train.fraction = 1, n.trees = 500, verbose = T )
p  <- predict(fit.gbm, test, n.trees = 500)
p <- expm1 (p)


submit  <- data.frame( datetime = test$datetime, count = p)
write.csv(submit, file= "~/Documents/Kaggle//Bike Sharing/submission1.csv", quote = F, row.names = F)

