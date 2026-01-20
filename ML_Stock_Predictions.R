library(tidyquant)
library(dplyr)
library(tidyr)
library(ggplot2)
library(TTR) #for SMA and RSI
library(zoo) #for rollapply
library(ROCR)
library(lars) 
library(rpart)
library(randomForest)
library(mgcv)
library(glmnet)



# Data prep
ticker <- "SPY"
start_date <- "2015-01-01"
end_date <- "2025-01-01"


# Data download
prices_raw <- tq_get(ticker, from = start_date, to = end_date)

head(prices_raw)

# Creating returns and target var
prices <- prices_raw %>%
  arrange(date) %>%
  mutate(return = log(adjusted/lag(adjusted)), # creating log returns
         up_tomorrow = ifelse(lead(return) > 0, 1, 0)) #if tomorrow is positive, then 1, 0 otherwise 
         
head(prices)
table(prices$up_tomorrow, useNA = "ifany")


# Making the indicators
prices_indicators <- prices %>% arrange(date) %>%
  mutate(
    lag_ret_1 = lag(return, 1),
    lag_ret_2 = lag(return, 2), #short term
    lag_ret_10 = lag(return, 10),
    lag_ret_30 = lag(return, 30), #medium term 
    
    sma_5 = SMA(adjusted, n=5), #simple moving average (5-day)
    sma_10 = SMA(adjusted, n=10),
    sma_30 = SMA(adjusted, n=30),
    
    dist_sma_5 = (adjusted/sma_5) - 1, # we normalize because the price is non-stationary (could be 200 early in the period but 500 at a later stage) - specifically needed for decision trees
    dist_sma_10 = (adjusted/sma_10) - 1,
    dist_sma_30 = (adjusted/sma_30) - 1,
    
    vol_10 = rollapply(return, width=10, FUN=sd, fill=NA, align="right"), #10-day rolling volatility 
    
    mom_10 = adjusted/lag(adjusted, 10) - 1, #a momentum indicator
    mom_30 = adjusted/lag(adjusted, 30) - 1,
    
    range_hl = (high - low)/adjusted, #intra-day range (normalized)
    co_diff = (close - open)/adjusted, #day-difference (normalized)
    
    vol_ma_20 = rollapply(volume, width=20, FUN=mean, fill=NA, align="right"), #20-day moving-average volume
    vol_rel = volume/vol_ma_20, #normalized volume measure
    log_vol_rel = log(vol_rel) #we transform in case of skewness
  ) %>% 
  select(date, symbol, up_tomorrow,
         lag_ret_1, lag_ret_2, lag_ret_10, lag_ret_30,
         dist_sma_5, dist_sma_10, dist_sma_30,
         vol_10, mom_10, mom_30,
         range_hl, co_diff, log_vol_rel) #we only select variables that are normalized or don't need normalization (non-stationarity of price)


model_data <- prices_indicators %>% drop_na() #we filter so our data contains no NAs
summary(model_data)



#Price and trend indicators
p_price <- ggplot(prices, aes(x=date, y=adjusted)) + geom_line() +
  labs(title="SPY Adjusted Close", x="Date", y="Adjusted Close")
p_price

dist_long <- prices_indicators %>%
  select(date, dist_sma_5, dist_sma_30) %>%
  pivot_longer(cols=c(dist_sma_5, dist_sma_30),
               names_to="indicator", values_to="value")

p_dist <- ggplot(dist_long, aes(x=date, y=value, color=indicator)) + geom_line() +
  labs(title="Distance to Moving Averages", x="Date", y="Deviation")
p_dist



#Splitting the data into train and test
n <- nrow(model_data)
train_size <- floor(0.85*n)
train_data <- model_data[1:train_size, ]
test_data <- model_data[(train_size+1):n, ]

x_train <- train_data %>% select(-date, -up_tomorrow) #a matrix with only predictor vars
y_train <- train_data$up_tomorrow #only dependent var

x_test <- test_data %>% select(-date, -up_tomorrow)
y_test <- test_data$up_tomorrow

eval_model <- function(pred_prob, y_test, model_name, threshold = 0.5) {
  #Log likelihood: how well predicted probs fit observed outcomes  
  LL <- sum(ifelse(y_test==1, log(pred_prob), log(1-pred_prob)))
  
  #Normalized deviance: model fit
  dev_norm <- -2*LL/length(y_test)
  
  #Confusion matrix
  cmatrix <- table(
    Predicted=factor(pred_prob>threshold, levels=c(FALSE, TRUE)), 
    Actual=factor(y_test, levels=c(0,1))) #setting levels because glmnet doesnt return 2x2 matrix
  
  #Accuracy: proportion of correctly classified obs.
  accuracy <- sum(diag(cmatrix))/sum(cmatrix)
  
  #Precision: how many obs predicted as positive are actually positive
  precision <- cmatrix[2,2]/ sum(cmatrix[2,])
  
  #Recall: how many of the actuall positives were identified correctly 
  recall <- cmatrix[2,2]/sum(cmatrix[,2])
  
  #F1: balance between precision and recall
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


#Logit Regression as baseline/benchmark
logit <- glm(up_tomorrow ~ ., 
             data=train_data %>% select(-date, -symbol), #we dont take the date because we dont treat it as a predictor
             family=binomial(link="logit"))
logit_pred_prob <- predict(logit, newdata=test_data %>% select(-date, -symbol), type="response") #get predicted probabilities
logit_pred_class <- ifelse(logit_pred_prob > 0.5,1,0) #turn predicted probabilities into 0/1


pred_obj_logit <- prediction(logit_pred_prob, as.numeric(y_test)) #put predictions and true outcomes in one object
auc_obj_logit <- performance(pred_obj_logit, "auc")@y.values[[1]]
auc_logit 

roc_perf_logit <- performance(pred_obj_logit, "tpr", "fpr")
plot(roc_perf_logit, main="ROC Curve - Logit")

results_logit <- eval_model(logit_pred_prob, y_test, "Logit") 
results_logit
     

#LASSO - we run logit here and use lasso for variable selection only as lasso uses ols and it cannot be used for a binary outcome
x_lasso <- model.matrix(up_tomorrow ~., data=train_data %>% select(-date, -symbol)) #required by lars (numeric matrix)
head(x_lasso)

x_lasso <- x_lasso[,-1] #drop intercept

lasso <- lars(x=x_lasso, y=y_train, trace=TRUE)
lasso
plot(lasso)

cv_lasso <- cv.lars(x=x_lasso, y=y_train, K=10) #cross validation
s_min_lasso <- cv_lasso$index[which.min(cv_lasso$cv)] #how strong the lasso penalty should be (s that minimizes CV error)
coef_lasso <- coef(lasso, s=s_min_lasso, mode="fraction") #the lasso coefs at the CV-optimal shrinkage level
sel_vars_lasso <- names(coef_lasso)[coef_lasso!=0] #take the names of the selected predictors
coef_lasso
sel_vars_lasso


f_lasso <- as.formula(paste("up_tomorrow ~ ", paste(sel_vars_lasso, collapse="+")))  #penalises all variables to 0 -> need to elaborate on that in the paper
logit_lasso <- glm(f_lasso, data=train_data %>% select(-date, -symbol), family=binomial(link="logit"))
lasso_pred_prob <- predict(logit_lasso, newdata=test_data %>% select(-date, -symbol), type="response")


results_lasso <- eval_model(lasso_pred_prob, y_test, "LASSO")
results_lasso




#LASSO2 - Glmnet
x_train_mat <- as.matrix(train_data %>% select(-date, -symbol, -up_tomorrow))
y_train_vec <- as.numeric(train_data$up_tomorrow)

x_test_mat <- as.matrix(test_data %>% select(-date, -symbol, -up_tomorrow))


set.seed(1234)
cv_lasso2 <- cv.glmnet(x=x_train_mat, y=y_train_vec, family="binomial", alpha=1, nfolds=10)
plot(cv_lasso2)

lasso_pred_prob2 <- predict(cv_lasso2, newx=x_test_mat, s="lambda.min", type="response")
results_lasso2 <- eval_model(as.numeric(lasso_pred_prob2), y_test, "LASSO-Glmnet")
results_lasso2

coef_lasso2 <- coef(cv_lasso2, s="lambda.min") #the lasso coefs at the CV-optimal shrinkage level
sel_vars_lasso2 <- names(coef_lasso2)[coef_lasso2!=0]
coef_lasso2
sel_vars_lasso2





#Decision Tree 
dec_tree <- rpart(as.factor(up_tomorrow) ~ ., data=train_data %>% select(-date, -symbol), method="class")
dec_tree_pred_prob <- predict(dec_tree, newdata=test_data %>% select(-date, -symbol), type="prob")[,2] #take the y=1 prob
head(dec_tree_pred_prob)
plot(dec_tree)
text(dec_tree) #todays return, adjusted close
print(dec_tree)
dec_tree$cptable
results_dec_tree <- eval_model(dec_tree_pred_prob, y_test, "Decision Tree")
results_dec_tree

#control
tree_ctrl <- rpart.control(cp=0.001, maxdepth=5, minsplit=20) #deeper decision tree fits the training data better but generalizes worse out of sample (overfitting)
dec_tree_ctrl <- rpart(as.factor(up_tomorrow) ~ ., data=train_data %>% select(-date, -symbol), method="class", control=tree_ctrl)
dec_tree_pred_prob_ctrl <- predict(dec_tree_ctrl, newdata=test_data %>% select(-date, -symbol), type="prob")[,2] 
plot(dec_tree_ctrl)
text(dec_tree_ctrl)
results_dec_tree_ctrl <- eval_model(dec_tree_pred_prob_ctrl, y_test, "Decision Tree")
results_dec_tree_ctrl
print(dec_tree_ctrl)
dec_tree_ctrl$cptable #for appendix



#Random Forest
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



#GAM
gam_formula <- as.formula(up_tomorrow ~ lag_ret_1 + lag_ret_2 + lag_ret_10 + lag_ret_30 + #linearly modeled: very noisy
                            s(dist_sma_5) + s(dist_sma_10) + s(dist_sma_30) + #non-linearly modeled: try to capture regimes
                            s(vol_10) + s(mom_10) + s(mom_30) +
                            s(range_hl) + s(co_diff) + s(log_vol_rel))

gam_logit <- gam(gam_formula, data=train_data %>% select(-date, -symbol), family=binomial(link="logit"))
summary(gam_logit)
gam_logit$converged #to check if gam found a solution

gam_pred_prob <- predict(gam_logit, newdata=test_data %>% select(-date, -symbol), type="response")
head(gam_pred_prob)

results_gam_logit <- eval_model(as.numeric(gam_pred_prob), y_test, "GAM-Logit")
results_gam_logit



#Naive benchmark that chooses unconditional prob of up-day (base-rate) (threshold = 0.5, mean >threshold)
naive_pred_prob <- rep(mean(y_train), length(y_test))
results_naive <- eval_model(naive_pred_prob, y_test, "Naive (base-rate)")
results_naive




#All results
results_all <- rbind(results_logit, results_lasso, results_lasso2, results_dec_tree, results_dec_tree_ctrl, results_rand_f, results_gam_logit, results_naive)
results_all 
# we chose a very narrow prediction horizon (next day). 
#It is well established in finance, that in short time-horizons stock returns are extremely noisy. As such, tomorrows price could be nothing different than a Drunkard's (Random) Walk.
#also the choice of train_data horizon will have an effect of the predictions
# how far back in time we should go in order to predict future's return is an interesting question
# if we go really far back, was the market/the stock the same as it is now...?



