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


model_data <- prices_indicators %>% #we filter so pur data contains no NAs
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



