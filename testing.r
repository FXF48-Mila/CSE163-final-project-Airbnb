# Liuyixin Shao
# CSE 163 AC
# This file contains all the testing in Rstudio for the final project
# This file mainly test dataframes used for research question 1 and 2.

library(tidyverse)
library(geojsonio)
library(compare)

# Q1:
nyc_df <- read.csv("testing_q1_r.csv")
nyc_neighborhood_df <- geojson_read("nyc-ny-neighborhood.geojson", what = 'sp')
nyc_neighborhood_df$neighbourhood = toupper(nyc_neighborhood_df$name)
test_q1_py <- read.csv("testing_q1_py.csv")

group_price_df <- nyc_df %>% 
  group_by(neighbourhood) %>% 
  summarize(airbnb_mean = mean(price))

group_sales_df <- nyc_df %>% 
  group_by(neighbourhood) %>% 
  summarize(sale_mean = mean(average_price))

group_df <- merge(group_price_df, group_sales_df, by = 'neighbourhood')

group_df['relative'] <- (group_df['airbnb_mean'] / 
  group_df ['sale_mean']) * 100000

test_q1_r <- merge(group_df, nyc_neighborhood_df, by = 'neighbourhood')

test_q1_r <- test_q1_r %>% select(neighbourhood, airbnb_mean, sale_mean, relative)

print(compare(test_q1_r, test_q1_py))

# Q2: 
geo_nyc_test <- read.csv("testing_q2_r.csv")
test_q2_py <- read.csv("testing_q2_py.csv")

test_q2_r <- geo_nyc_test %>% 
  arrange(desc(number_of_reviews)) %>% 
  slice_head(n = 20)

print(compare(test_q2_r, test_q2_py))