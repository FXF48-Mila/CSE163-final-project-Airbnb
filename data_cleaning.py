"""
Liuyixin Shao
CSE 163 AC
This file did all the data cleaning processes for the final project,
including merging datasets, constructing geoDataFrames, and constructing
grouped datasets.
"""


import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import numpy as np


def float_to_int(float_num):
    """
    Takes a number in float format
    Returns the number in integer format
    """
    return int(float_num)


def delete_comma(str_comma):
    """
    Takes a number in string format
    Deletes the commas that seperate every three digits
    Returns the number in integer format
    """
    save = str_comma.split(',')
    result = ''.join(save)
    return int(result)


def combine_five_dataset(brox_price, brooklyn_price, manhattan_price,
                         queens_price, staten_island_price):
    """
    Takes the 5 converted CSV datasets from the NYC Department of Finance
    Concatenates the 5 datasets together, deletes the rows with missing
    values, and calculates the total price of each type of housing in each
    neighborhood
    Returns the merged dataset with total price calculated
    """
    combine_price = pd.concat([brox_price, brooklyn_price, manhattan_price,
                               queens_price, staten_island_price])
    combine_price = combine_price[['BOROUGH', 'NEIGHBORHOOD',
                                   'NUMBER OF SALES', 'AVERAGE SALE PRICE']]
    combine_price = combine_price.dropna()
    combine_price['NUMBER OF SALES'] = combine_price['NUMBER OF SALES'] \
        .apply(float_to_int)
    combine_price['AVERAGE SALE PRICE'] = \
        combine_price['AVERAGE SALE PRICE'].apply(delete_comma)
    combine_price['total_price'] = combine_price['AVERAGE SALE PRICE'] \
        * combine_price['NUMBER OF SALES']
    return combine_price


def average_sale_price(combine_price):
    """
    Takes a merged dataset with total price calculated
    Calculates the average sales price in each neighborhood ignore room type
    Returns a dataset including neighborhoods in New York and the average
    sales price (ignore room type) of the properties in each of these
    neighborhoods
    """
    price_df = combine_price.groupby(['NEIGHBORHOOD'],
                                     as_index=False)['total_price'].sum()
    num_df = combine_price.groupby(['NEIGHBORHOOD'],
                                   as_index=False)['NUMBER OF SALES'].sum()
    average_price_nb_df = price_df.merge(num_df, left_on='NEIGHBORHOOD',
                                         right_on='NEIGHBORHOOD', how='outer')
    average_price_nb_df['average_price'] = \
        average_price_nb_df['total_price'] \
        / average_price_nb_df['NUMBER OF SALES']
    return average_price_nb_df


def check_diff(airbnb_nyc, average_price_nb_df):
    """
    Takes the New York City Airbnb Open Data and the dataset containing the
    average sale price (ignore room type) for each neighborhood in New York
    City
    Prints the neighborhood names' differences in these datasets
    """
    # check and replace name of the same neighborhood in different datasets
    airbnb_nyc['neighbourhood'] = airbnb_nyc['neighbourhood'].str.upper()
    series_nb_in_airbnb_df = airbnb_nyc['neighbourhood'].unique()
    # find the difference
    s1 = set(series_nb_in_airbnb_df)
    s2 = set(average_price_nb_df.neighborhood)
    difference = s1.difference(s2)
    print(sorted(difference))
    # By looking up the generated differences manually,
    # there is no difference between neighborhood names that I need to replace


def merge_dataframe(airbnb_nyc, average_price_nb_df):
    """
    Takes the New York City Airbnb Open Data and the dataset containing the
    average sale price (ignore room type) for each neighborhood in New York
    City
    Merges the two datasets by neighborhood
    Returns a merged dataset with 11 useful columns for the final project
    analysis as a pandas DataFrame
    """
    # merge the datasets together
    airbnb_nyc['neighbourhood'] = airbnb_nyc['neighbourhood'].str.upper()
    nyc_df = airbnb_nyc.merge(average_price_nb_df, left_on='neighbourhood',
                              right_on='NEIGHBORHOOD')
    # choose useful columns
    nyc_df = nyc_df[['name', 'neighbourhood_group', 'neighbourhood',
                     'room_type', 'price', 'minimum_nights',
                     'number_of_reviews', 'reviews_per_month',
                     'average_price', 'longitude', 'latitude']]
    return nyc_df


def to_geo_df(nyc_df):
    """
    Takes a merged pandas DataFrame
    Changes the latitude and longitude in the dataframe into points that
    GeoDataFrame can read, then adds the points to the new column in the
    dataframe called "coordinates"
    Returns the dataframe with coordinate information as a GeoDataFrame
    """
    # change the latitude and longitude into points that GeoDataFrame can read
    nyc_df['coordinates'] = [Point(long, lat) for long, lat in
                             zip(nyc_df['longitude'], nyc_df['latitude'])]
    # change the nyc_df DataFrame into a GeoDataFrame
    geo_nyc_df = gpd.GeoDataFrame(nyc_df, geometry='coordinates')
    return geo_nyc_df


def group_nb_df(nyc_df, nyc_neighborhood_df):
    """
    Takes the merged dataset of Airbnbs in New York City and the geojson
    dataset with geometry of the shapes of neighborhoods in New York City
    Returns a GeoDataFrame that groups by neighborhoods and contains the
    mean airbnb price, the mean property sale price, and the relative
    price in each neighborhood in New York City
    Note: The relative price is derived by dividing the average Airbnb price
    by the average property sale price, then multiplied by 10,000
    """
    # For question 1:
    nyc_neighborhood_df.rename(columns={'name': 'Neighborhood'}, inplace=True)
    nyc_neighborhood_df['Neighborhood'] = \
        nyc_neighborhood_df['Neighborhood'].str.upper()
    group_df = nyc_df.groupby(['neighbourhood'], as_index=False).agg(
        airbnb_mean=('price', 'mean'),
        sale_mean=('average_price', 'mean')
    )
    group_df['relative'] = (group_df['airbnb_mean']
                            / group_df['sale_mean']) * 100000
    geo_group_df = group_df.merge(nyc_neighborhood_df,
                                  left_on='neighbourhood',
                                  right_on='Neighborhood')
    geo_group_df = gpd.GeoDataFrame(geo_group_df, geometry='geometry')
    return geo_group_df


def c_filter_data(filtered_data):
    """
    Takes a filtered dataframe used for machine learning
    Split the price in the dataframe into 6 ranges:
    0-50, 50-100, 100-150, 150-200, 200-300, 300+
    (for each range, the number to the left of the range is included
    and the number to the right is not included)
    Return the dataframe with splitted price
    """
    cond = [(filtered_data['price'] >= 0) & (filtered_data['price'] < 50),
            (filtered_data['price'] >= 50) & (filtered_data['price'] < 100),
            (filtered_data['price'] >= 100) & (filtered_data['price'] < 150),
            (filtered_data['price'] >= 150) & (filtered_data['price'] < 200),
            (filtered_data['price'] >= 200) & (filtered_data['price'] < 300),
            (filtered_data['price'] >= 300)]
    choices = ['0-50', '50-100', '100-150', '150-200', '200-300', '300+']
    class_filtered_data = filtered_data.copy()
    class_filtered_data['price_range'] = np.select(cond, choices)
    class_filtered_data = class_filtered_data.loc[:, class_filtered_data.
                                                  columns != 'price']
    return class_filtered_data
