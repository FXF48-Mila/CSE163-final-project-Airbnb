"""
Liuyixin Shao
CSE 163 AC
This file contains all the testing in Visual Studio Code
for the final project, including loading all files for testing,
exporting cvs files from Visual Studio Code to RStudio, and
building a smaller testing file and test the function used in question 4.
Other testing are implemented in the file named testing.r
"""


import final
import pandas as pd
import geopandas as gpd
import data_cleaning


AIRBNB_NYC = './AB_NYC_2019.csv'
BROX_PRICE = './mutated_2021_bronx_sales_prices.csv'
BROOKLYN_PRICE = './mutated_2021_brooklyn_sales_prices.csv'
MANHATTAN_PRICE = './mutated_2021_manhattan_sales_prices.csv'
QUEENS_PRICE = './mutated_2021_queens_sales_prices.csv'
STATEN_ISLAND_PRICE = './mutated_2021_staten_island_sales_prices.csv'
MAP_NYC = './nyc-ny-neighborhood.geojson'


def main():
    airbnb_nyc = pd.read_csv(AIRBNB_NYC)
    brox_price = pd.read_csv(BROX_PRICE)
    brooklyn_price = pd.read_csv(BROOKLYN_PRICE)
    manhattan_price = pd.read_csv(MANHATTAN_PRICE)
    queens_price = pd.read_csv(QUEENS_PRICE)
    staten_island_price = pd.read_csv(STATEN_ISLAND_PRICE)
    nyc_neighborhood_df = gpd.read_file(MAP_NYC)
    combine_price = data_cleaning.combine_five_dataset(brox_price,
                                                       brooklyn_price,
                                                       manhattan_price,
                                                       queens_price,
                                                       staten_island_price)
    average_price_nb_df = data_cleaning.average_sale_price(combine_price)
    # Test 1:
    nyc_df = data_cleaning.merge_dataframe(airbnb_nyc, average_price_nb_df)
    # Export the nyc_df for testing question 1:
    nyc_df.to_csv('./testing_q1_r.csv', index=False)
    geo_nyc_df = data_cleaning.to_geo_df(nyc_df)
    geo_group_df = data_cleaning.group_nb_df(nyc_df,
                                             nyc_neighborhood_df)
    nyc_df_test = geo_group_df[['neighbourhood', 'airbnb_mean',
                                'sale_mean', 'relative']]
    # Export the filtered geo_group_df for testing question 1:
    nyc_df_test.to_csv('./testing_q1_py.csv', index=False)

    # Test 2:
    geo_nyc_test = geo_nyc_df[['neighbourhood', 'number_of_reviews']]
    # Export the filtered geo_nyc_df for testing question 2:
    geo_nyc_test.to_csv('./testing_q2_r.csv', index=False)
    geo_nyc_20 = geo_nyc_df.nlargest(20, 'number_of_reviews')
    geo_nyc_20_test = geo_nyc_20[['neighbourhood', 'number_of_reviews']]
    # Export the filtered geo_nyc_20 for testing question 2:
    geo_nyc_20_test.to_csv('./testing_q2_py.csv', index=False)

    # Test 4:
    # create a smaller dataframe
    nyc_df_name_test = nyc_df.loc[[7, 9, 48, 148, 2000, 2022,
                                   248, 448, 548, 600],
                                  ['name']]
    print(nyc_df_name_test)
    final.q4_count(nyc_df_name_test, './q4_frq_test.png', test=True)


if __name__ == '__main__':
    main()
