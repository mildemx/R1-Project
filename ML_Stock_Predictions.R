# Data prep
ticker <- "SPY"
start_date <- "2015-01-01"
end_date <- "2025-01-01"

# install.packages(c("tidyquant", "dplyr"))
library(tidyquant)
library(dplyr)

# Data download
prices_raw <- tq_get(ticker, from = start_date, to = end_date)

head(prices_raw)

# Creating returns and target var
prices <- prices_raw %>%
  arrange(date) %>%
  mutate(return = log(close/lag(close)), # creating log returns
         up_tomorrow = ifelse(lead(return) > 0, 1, 0)) #if tomorrow is positive, then 1, 0 otherwise 
         
head(prices)
table(prices$up_tomorrow, useNA = "ifany")


# Making the indicators
library(TTR) #for SMA and RSI
library(zoo) #for rollapply

prices_indicators <- prices %>%
  mutate(
    lag_ret_1 = lag(return, 1),
    lag_ret_2 = lag(return, 2),
    sma_5 = SMA(close, n=5), #simple moving average (5-day)
    sma_10 = SMA(close, n=10),
    vol_10 = rollapply(return, width=10, FUN=sd, fill=NA, align="right"), #10-day rolling volatility 
    mom_10 = close/lag(close, 10) - 1, #a momentum indicator
    range_hl = high - low, #intraday range
    co_diff = close - open,
    log_volume = log(volume+1)) #we transform so values are not so extreme


model_data <- prices_indicators %>% #we filter so our data contains no NAs
  filter(!is.na(up_tomorrow),
         !is.na(lag_ret_1),
         !is.na(lag_ret_2),
         !is.na(sma_5),
         !is.na(sma_10),
         !is.na(vol_10),
         !is.na(mom_10))
summary(model_data)


#Splitting the data into train and test
n <- nrow(model_data)
train_size <- floor(0.85*n)
train_data <- model_data[1:train_size, ]
test_data <- model_data[(train_size+1):n, ]

x_train <- train_data %>% select(-date, -up_tomorrow) #create a matrix with only predictor vars
y_train <- train_data$up_tomorrow #only dependent var

x_test <- test_data %>% select(-date, -up_tomorrow)
y_test <- test_data$up_tomorrow




#Logit Regression as baseline/benchmark
logit <- glm(up_tomorrow ~ ., 
             data=train_data %>% select(-date, -symbol), #we dont take the date because we dont treat it as a predictor
             family=binomial(link="logit"))
logit_pred_prob <- predict(logit, newdata=test_data %>% select(-date, -symbol), type="response") #get predicted probabilities
logit_pred_class <- ifelse(logit_pred_prob > 0.5,1,0) #turn predicted probabilities into 0/1

library(ROCR)
pred_obj_logit <- prediction(logit_pred_prob, as.numeric(y_test)) #put predictions and true outcomes in one object
auc_obj_logit <- performance(pred_obj_logit, "auc")@y.values[[1]]
auc_logit 

roc_perf_logit <- performance(pred_obj_logit, "tpr", "fpr")
plot(roc_perf_logit, main="ROC Curve - Logit")

LL_logit <- sum(ifelse(y_test==1, log(logit_pred_prob), log(1-logit_pred_prob))) #Log likelihood: how well predicted probs fit observed outcomes  
dev_norm_logit <- -2*LL_logit / length(y_test) #Norm. deviance: model fit
threshold_logit <- 0.5
cmatrix_logit <- table(Predicted=logit_pred_prob > threshold_logit, Actual=y_test) #Confusion matrix
accuracy_logit <- sum(diag(cmatrix_logit)/sum(cmatrix_logit)) #Accuracy: proportion of correctly classified obs.
precision_logit <- cmatrix_logit[2,2]/sum(cmatrix_logit[2,]) #Precision: how many obs predicted as positive are actually positive
recall_logit <- cmatrix_logit[2,2]/sum(cmatrix_logit[,2]) #Recall: how many of the actuall positives were identified correctly 
f1_logit <- 2*precision_logit*recall_logit/(precision_logit+recall_logit) #F1: balance between precision and recall

results_logit <- data.frame(Model="Logit", 
                            AUC=auc_logit, 
                            Accuracy=accuracy_logit,
                            Precision=precision_logit, 
                            Recall=recall_logit, 
                            F1=f1_logit, 
                            Dev_Norm=dev_norm_logit)   
results_logit
     
     
     

#LASSO - we run logit here and use lasso for variable selection only as lasso uses ols and it cannot be used for a binary outcome
library(lars)
x_lasso <- model.matrix(up_tomorrow ~., data=train_data %>% select(-date, -symbol)) #required by lars (numeric matrix)
head(x_lasso)

x_lasso <- x_lasso[,-1] #drop intercept

lasso <- lars(x=x_lasso, y=y_train, trace=TRUE)
lasso
plot(lasso)

cv_lasso <- cv.lars(x=x_lasso, y=y_train, K=100) #cross validation
s_min_lasso <- cv_lasso$index[which.min(cv_lasso$cv)] #how strong the lasso penalty should be (s that minimizes CV error)
coef_lasso <- coef(lasso, s=s_min_lasso, mode="fraction") #the lasso coefs at the CV-optimal shrinkage level
sel_vars_lasso <- names(coef_lasso)[coef_lasso!=0] #take the names of the selected predictors

f_lasso <- as.formula(paste("up_tomorrow ~", paste(sel_vars_lasso, collapse="+"))) 
logit_lasso <- glm(f_lasso, data=train_data %>% select(-date, -symbol), family=binomial(link="logit"))
lasso_pred_prob <- predict(logit_lasso, newdata=test_data %>% select(-date, -symbol), type="response")


eval_model <- function(pred_prob, y_test, model_name, threshold = 0.5) {
  #LL
  LL <- sum(ifelse(y_test==1, log(pred_prob), log(1-pred_prob)))
  
  #Normalized deiance
  dev_norm <- -2*LL/length(y_test)
  
  #Confusion matrix
  cmatrix <- table(Predicted=pred_prob>threshold, Actual=y_test)
  
  #Accuracy
  accuracy <- sum(diag(cmatrix))/sum(cmatrix)
  
  #Precision, Recall, F1
  precision <- cmatrix[2,2]/ sum(cmatrix[2,])
  recall <- cmatrix[2,2]/sum(cmatrix[,2])
  f1 <- 2*precision*recall/(precision+recall)
  
  #AUC
  auc <- performance(prediction(pred_prob, y_test), "auc")@y.values[[1]]
  
  #Output
  data.frame(Model=model_name, 
             AUC=auc, 
             Accuracy=accuracy,
             Precision=precision, 
             Recall=recall, 
             F1=f1, 
             Dev_Norm=dev_norm)   
} #function to give the comparison metrics

results_lasso <- eval_model(lasso_pred_prob, y_test, "LASSO")
results_lasso





#Decision Tree
library(rpart)

dec_tree <- rpart(as.factor(up_tomorrow) ~ ., data=train_data %>% select(-date, -symbol), method="class")
dec_tree_pred_prob <- predict(dec_tree, newdata=test_data %>% select(-date, -symbol), type="prob")[,2] #take the y=1 prob
head(dec_tree_pred_prob)
plot(dec_tree)
text(dec_tree) #todays return, adjusted close
results_dec_tree <- eval_model(dec_tree_pred_prob, y_test, "Decision Tree")
results_dec_tree



#Random Forest
library(randomForest)
set.seed(1234)

rand_f <- randomForest(x=train_data %>% select(-date, -symbol, -up_tomorrow), #predictors
                   y=as.factor(train_data$up_tomorrow), #outcome
                   ntree=33,
                   nodesize=7, #min number of obs at terminal node
                   importance=TRUE)
rand_f_pred_prob <- predict(rand_f, newdata=test_data %>% select(-date, -symbol, -up_tomorrow), type="prob")[,2] #get probabs
head(rand_f_pred_prob)

rand_f_import <- importance(rand_f, type=1)
head(rand_f_import)
varImpPlot(rand_f, type=1) #to see how much each variable helps with prediction
     
results_rand_f <- eval_model(rand_f_pred_prob, y_test, "Random Forest")
results_rand_f




#All results
results_all <- rbind(results_logit, results_lasso, results_dec_tree, results_rand_f)
results_all




