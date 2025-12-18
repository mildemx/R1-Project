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
