"""
Liuyixin Shao
CSE 163 AC
This file contains all the functions for the final project,
including constructing maps, building, training, and testing
machine learning models, and ploting charts for analysis.
"""


import warnings
from warnings import simplefilter
import data_cleaning
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
from shapely.errors import ShapelyDeprecationWarning
# The FutureWarning is caused by a package that is inside the library
# xgboost which uses the old version of pandas
# Therefore, here, I simply ignore all FutureWarnings
# However, because I have to follow the flake8 test and put all the import
# functions before the actual code to deal with warning, you can still
# see the warning from importing xgboost. I have verified this with Wen.
simplefilter(action='ignore', category=FutureWarning)
# The UserWarning is caused by the classifier in xgboost is compatible
# with testing methods in sklearn
# Therefore, here, I simply ignore all UserWarnings
warnings.filterwarnings("ignore", category=UserWarning)
# The ShapelyDeprecationWarning is caused by the different version of
# shapely in the laptop.
# I have already checked with Ryan that my laptop actually contains
# the latest version of all the libries that cause this warning.
# I have already talked with Wen and finally we decided to ignore this
# warning.
warnings.filterwarnings("ignore", category=ShapelyDeprecationWarning)


sns.set()

# read the daatframes
AIRBNB_NYC = './AB_NYC_2019.csv'
BROX_PRICE = './mutated_2021_bronx_sales_prices.csv'
BROOKLYN_PRICE = './mutated_2021_brooklyn_sales_prices.csv'
MANHATTAN_PRICE = './mutated_2021_manhattan_sales_prices.csv'
QUEENS_PRICE = './mutated_2021_queens_sales_prices.csv'
STATEN_ISLAND_PRICE = './mutated_2021_staten_island_sales_prices.csv'
MAP_NYC = './nyc-ny-neighborhood.geojson'


def q1_mean_price(geo_group_df, nyc_neighborhood_df):
    """
    Takes the grouped Airbnb dataframe and the geojson dataframe with
    neighborhood shapes in New York City
    Plots 2 charts showing the distribution of the average Airbnb price
    (chart1) and the average property sale price (chart2) in each
    neighborhood side-by-side
    Includes legends in each chart
    Titles the first chart as "Distribution of Average Airbnb Price in
    NYC by Neighborhood" and the second chart as "Distribution of Average
    Property Sale Price in NYC by Neighborhood"
    Saves the plot as ./q1_mean.png
    """
    # mean airbnb price distribution by neighborhood
    fig, [ax1, ax2] = plt.subplots(ncols=2, figsize=(20, 10))
    nyc_neighborhood_df.plot(color='#AAAAAA', ax=ax1)
    geo_group_df.plot(column='airbnb_mean', legend=True, ax=ax1)
    ax1.set_title('Distribution of Average Airbnb Price in '
                  'NYC by Neighborhood')
    # mean sale price distribution by neighborhood
    nyc_neighborhood_df.plot(color='#AAAAAA', ax=ax2)
    geo_group_df.plot(column='sale_mean', legend=True, ax=ax2)
    ax2.set_title('Distribution of Average Property Sale '
                  'Price in NYC by Neighborhood')
    plt.savefig('./q1_mean.png')


def q1_relative_price(geo_group_df, nyc_neighborhood_df):
    """
    Takes the grouped Airbnb dataframe and the geojson dataframe with
    neighborhood shapes in New York City
    Plots the distribution of the relative price in each neighborhood
    in New York City
    The relative price is derived by dividing the average Airbnb price
    by the average property sale price, then multiplied by 100000
    Includes legends in each chart
    Titles the chart as "Relative Average Airbnb Price to Property
    Sale Price in NYC by neighborhood (times 100000)"
    Saves the plot as ./q1_relative.png
    """
    fig, ax = plt.subplots(1, figsize=(10, 10))
    nyc_neighborhood_df.plot(color='#AAAAAA', ax=ax)
    geo_group_df.plot(column='relative', legend=True, ax=ax)
    plt.title('Relative Average Airbnb Price to Property Sale Price in'
              ' NYC by neighborhood (times 100000)')
    plt.savefig('./q1_relative.png')


def q2_pop(nyc_neighborhood_df, geo_nyc_20):
    """
    Takes the the geojson dataframe with neighborhood shapes in New York City
    and the sliced dataframe with top 20 total reviews
    Plots the distribution of the Airbnbs with the top 20 total reviews by
    coloring the exact location of each Airbnb in orange and the neighborhood
    each Airbnb belongs to in purple
    Titles the chart as "Distribution of the Top 20 Most Popular Airbnbs"
    Save the plot as ./q2_pop.png
    """
    geo_nyc_20 = geo_nyc_20.set_crs(epsg=4326)
    pop_nb = gpd.sjoin(nyc_neighborhood_df, geo_nyc_20,
                       how='inner', predicate='intersects')
    fig, ax = plt.subplots(1, figsize=(15, 10))
    nyc_neighborhood_df.plot(color='#AAAAAA', ax=ax)
    pop_nb.plot(ax=ax, color='#3c1361')
    geo_nyc_20.plot(ax=ax, color='#FFA500', markersize=10)
    plt.title('Distribution of the Top 20 Most Popular Airbnbs')
    plt.savefig('./q2_pop.png')


def q2_ployly_table(geo_nyc_20):
    """
    Takes the sliced dataframe with top 20 total reviews
    Plots a table showing the name, the neighborhood group, the
    neighborhood, the room type, the Airbnb price, and the number
    of reviews of the top 20 Airbnbs
    Titles the table as "The 20 Most Popular Airbnbs Stats"
    Note: to save the plot generated from plotly, click the camera
    label on the top right of the popping website
    """
    fig = go.Figure(data=[go.Table(
        header=dict(values=['Name', 'Neighbourhood Group',
                            'Neighbourhood', 'room type',
                            'Airbnb price', 'Number of Reviews'],
                    fill_color='paleturquoise',
                    align='left'),
        cells=dict(values=[geo_nyc_20.name, geo_nyc_20.neighbourhood_group,
                           geo_nyc_20.neighbourhood, geo_nyc_20.room_type,
                           geo_nyc_20.price, geo_nyc_20.number_of_reviews],
                   fill_color='lavender', align='left'))
    ])
    fig.update_layout(
        title_text="The 20 Most Popular Airbnbs Stats",
    )
    fig.show()


def q3_decision_tree_valid(features_train, labels_train, features_valid,
                           labels_valid):
    """
    Takes the training and validation dataframes of features and labels
    (the regression version)
    Train a DecisionTreeRegressor model with max_depth equals 2, 3, 4, 5, 6
    Print the mean squared error, the mean absolute error, and the testing
    score of the model with different maximum depths using the validation
    datasets
    Note: I set the random state equals to 1 in order to make comparsions
    """
    # build model
    print('DecisionTreeRegressor')
    for max_depth in [2, 3, 4, 5, 6]:
        model = DecisionTreeRegressor(max_depth=max_depth,
                                      random_state=1)
        model.fit(features_train, labels_train)
        # cauculate the stats
        test_predictions = model.predict(features_valid)
        test_error = mean_squared_error(labels_valid, test_predictions)
        abs_error = mean_absolute_error(labels_valid, test_predictions)
        score = model.score(features_valid, labels_valid)
        print("Max Depth: %d  \t Mean Squared Error:  %.2f  \
            Mean abs Error: %.2f  \t Testing score: %.4f" % (max_depth,
                                                             test_error,
                                                             abs_error,
                                                             score))
    # we can see when max depth = 5, we get the smallest mean square
    # error and largest testing score, but the error is still huge


def q3_random_forest_valid(features_train, labels_train, features_valid,
                           labels_valid):
    """
    Takes the training and validation dataframes of features and labels
    (the regression version)
    Train a RandomForestRegressor model with n_estimators equals to
    100, 1000, 1500, 1700, and 2000
    Print the mean squared error, the mean absolute error, and the testing
    score of the model with different number of trees to be used in the forest
    using the validation datasets
    Note: I set the random state equals to 1 and max_depth equals to 5
    in order to make comparsions
    """
    # build model
    print('RandomForestRegressor')
    for n_estimators in [100, 1000, 1500, 1700, 2000]:
        model = RandomForestRegressor(n_estimators=n_estimators,
                                      max_depth=5,
                                      random_state=1)
        model.fit(features_train, labels_train)
        # calculate the stats
        test_predictions = model.predict(features_valid)
        test_error = mean_squared_error(labels_valid, test_predictions)
        abs_error = mean_absolute_error(labels_valid, test_predictions)
        score = model.score(features_valid, labels_valid)
        print("N_estimators: %d  \t Mean Squared Error:  %.2f  \
            Mean abs Error: %.2f  \t Testing score: %.4f" % (n_estimators,
                                                             test_error,
                                                             abs_error,
                                                             score))
    # This model is worse, but at least we know a proper n_estimators
    # for the model.


def q3_boost_valid(features_train, labels_train, features_valid,
                   labels_valid):
    """
    Takes the training and validation dataframes of features and labels
    (the regression version)
    Train a XGBRegressor model with learning_rate equals to
    0.003, 0.002, 0.0017, 0.0015, 0.0013, 0.001
    Print the mean squared error, the mean absolute error, and the testing
    score of the model with different tuning parameters using the
    validation datasets
    Note: I set the random state equals to 1, the max_depth equals to 5,
    and the n_estimators equals to 1500 in order to make comparsions
    """
    # build model
    print('XGBRegressor')
    for learning_rate in [0.003, 0.002, 0.0017, 0.0015, 0.0013, 0.001]:
        model = XGBRegressor(n_estimators=1500, learning_rate=learning_rate,
                             max_depth=5, random_state=1)
        model.fit(features_train, labels_train,
                  eval_set=[(features_valid, labels_valid)],
                  verbose=False)
        # cauculate the stats
        test_predictions = model.predict(features_valid)
        test_error = mean_squared_error(labels_valid, test_predictions)
        abs_error = mean_absolute_error(labels_valid, test_predictions)
        score = model.score(features_valid, labels_valid)
        print("Learning rate: %.4f  \t Mean Squared Error:  %.2f  \
            Mean abs Error: %.2f  \t Testing score: %.4f" % (learning_rate,
                                                             test_error,
                                                             abs_error,
                                                             score))


def q3_plot():
    """
    Plots a chart with 3 tables as subplots showing the changing
    hyper-parameters, the mean square error, the mean absolute error,
    and the testing score of each generated machine learning model (regressor)
    Color the row with selected hyper-parameters in light green
    Titles the first table as "Validate Max Depth in
    DecisionTreeRegressor",
    the second table as "Validate N Estimators in RandomForestRegressor",
    and the third table as "Validate Learning Rate in XGBRegressor"
    Titles the chart as "Table Comparing the 3 Regression Models"
    Note: to save the plot generated from plotly, click the camera
    label on the top right of the popping website
    """
    select_color = 'lightgreen'
    other = 'lavender'
    fig = make_subplots(
        rows=3, cols=1,
        specs=[[{"type": "table"}],
               [{"type": "table"}],
               [{"type": "table"}]],
        subplot_titles=['Validate Max Depth in DecisionTreeRegressor',
                        'Validate N Estimators in RandomForestRegressor',
                        'Validate Learning Rate in XGBRegressor']
    )
    # DecisionTreeRegressor
    fig.add_trace(go.Table(
        header=dict(
            values=["Max Depth", "Mean Squared Error",
                    "Mean Absolute Error",  "Testing Score"],
            fill_color='paleturquoise',
            align="left"
        ),
        cells=dict(
            values=[[2, 3, 4, 5, 6],
                    [18276.07, 17390.10, 17274.70, 17228.92, 55483.89],
                    [58.21, 55.54, 54.80, 54.42, 58.64],
                    [0.1587, 0.1995, 0.2048, 0.2069, -1.5541]],
            fill_color=[[other, other, other, select_color, other]*5],
            align="left")
    ),
        row=1, col=1
    )
    # RandomForestRegressor
    fig.add_trace(go.Table(
        header=dict(
            values=["N Estimators", "Mean Squared Error",
                    "Mean Absolute Error",  "Testing Score"],
            fill_color='paleturquoise',
            align="left"
        ),
        cells=dict(
            values=[[100, 1000, 1500, 1700, 2000],
                    [20757.49, 19264.79, 19105.08, 19148.33, 19125.55],
                    [55.17, 54.89, 54.88, 54.91, 54.90],
                    [0.0445, 0.1132, 0.1205, 0.1185, 0.1196]],
            fill_color=[[other, other, select_color, other, other]*5],
            align="left")
    ),
        row=2, col=1
    )
    # XGBRegressor
    fig.add_trace(go.Table(
        header=dict(
            values=["Learning Rate", "Mean Squared Error",
                    "Mean Absolute Error",  "Testing Score"],
            fill_color='paleturquoise',
            align="left"
        ),
        cells=dict(
            values=[[0.0030, 0.0020, 0.0017, 0.0015, 0.0013, 0.0010],
                    [18927.76, 17771.83, 17718.76, 17727.16, 17757.21,
                     18064.79],
                    [54.74, 52.54, 51.37, 50.54, 49.73, 49.52],
                    [0.1287, 0.1819, 0.1844, 0.1840, 0.1826, 0.1684]],
            fill_color=[[other, other, other, other, select_color, other]*6],
            align="left")
    ),
        row=3, col=1
    )

    fig.update_layout(
        title_text="Table Comparing the 3 Regression Models",
    )
    fig.show()


def q3_final_model(features_train, labels_train, features_test,
                   labels_test, features_valid, labels_valid):
    """
    Takes the training, validation, and testing dataframes of features
    and labels (the regression version)
    Train a XGBRegressor model with the proper hyper-parameters I chose
    and fit the model with validation datasets as evalution for the model
    Print the mean squared error, the mean absolute error, and the testing
    score of the model using the testing datasets
    Then,
    Plots a chart with 2 tables as subplots showing the mean square error,
    the mean absolute error, and the testing score of the final chosen
    XGBRegressor model and its actual and prediction performance in the first
    30 rows of the testing data
    Titles the first table as "Testing Statistics of the XGBRegressor Model",
    and the second table as "Actual and Predicted Airbnb Price Comparison",
    Note: to save the plot generated from plotly, click the camera
    label on the top right of the popping website
    """
    print('Chosen XGBRegressor:')
    model = XGBRegressor(n_estimators=1500, learning_rate=0.0013,
                         max_depth=5)
    model.fit(features_train, labels_train,
              eval_set=[(features_valid, labels_valid)],
              verbose=False)
    # cauculate the mean squared error
    test_predictions = model.predict(features_test)
    test_error = mean_squared_error(labels_test, test_predictions)
    abs_error = mean_absolute_error(labels_test, test_predictions)
    score = model.score(features_test, labels_test)
    print("Mean Squared Error:  %.2f  \
        Mean Absolute Error: %.2f  \t Testing Score: %.4f" % (test_error,
                                                              abs_error,
                                                              score))

    error = pd.DataFrame({'actual': np.array(labels_test).flatten(),
                          'predict': test_predictions.flatten()})
    show = error.head(30)
    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{"type": "table"}, {"type": "table"}]],
        subplot_titles=['Testing Statistics of the XGBRegressor Model',
                        'Actual and Predicted Airbnb Price Comparison']
    )
    fig.add_trace(go.Table(
        header=dict(
            values=["Mean Squared Error",
                    "Mean Absolute Error",  "Testing Score"],
            fill_color='paleturquoise',
            align="left"
        ),
        cells=dict(
            values=[82494.84, 57.33, 0.0412],
            fill_color='lavender',
            align="left")
    ),
        row=1, col=1
    )
    fig.add_trace(go.Table(
        header=dict(values=['Actual Values', 'Predicted Values'],
                    fill_color='paleturquoise',
                    align='left'),
        cells=dict(values=[show.actual, show.predict],
                   fill_color='lavender', align='left')
    ),
        row=1, col=2
    )
    fig.update_layout(
        title_text="Performance of the Selected XGBRegressor Model"
    )
    fig.show()


def q3_plot_price(filtered_data):
    """
    Takes the filtered dataset used for machine learning
    Plots a density chart showing the distribution of Airbnb prices
    Titles the plot as "Distribution of Airbnb Prices"
    Saves the plot as ./q3_plot_price.png
    """
    # I filtered the data because I want to exclude the extreme outliers
    # and make the distribution sharper.
    filtered_data = filtered_data[filtered_data['price'] <= 500].copy()
    # Here, I have to use the displot function to build the density plot
    # because on last Thursday Ryan and I found the kdeplot in seaborn
    # crashed...
    sns.displot(data=filtered_data, x="price", kind='kde')
    plt.title('Distribution of Airbnb Prices')
    plt.savefig('./q3_plot_price.png', bbox_inches='tight')


def q3_c_decision_tree_valid(c_features_train, c_labels_train,
                             c_features_valid, c_labels_valid):
    """
    Takes the training and validation dataframes of features and labels
    (the classification version)
    Train a DecisionTreeClassifier model with max_depth equals 2, 5, 6, 7, 8
    Print the accuracy score of the model with different maximum depths
    using the validation datasets
    Note: I set the random state equals to 1 in order to make comparsions
    """
    # build model
    print('DecisionTreeClassifier')
    for max_depth in [2, 5, 6, 7, 8]:
        model = DecisionTreeClassifier(max_depth=max_depth,
                                       random_state=1)
        model.fit(c_features_train, c_labels_train)
        # cauculate the stats
        train_acc = model.score(c_features_train, c_labels_train)
        test_acc = model.score(c_features_valid, c_labels_valid)
        print("Max Depth: %d  \t Train Accuracy:  %.4f  \
            Test Accuracy: %.4f" % (max_depth, train_acc, test_acc))
    # we can see the classification models perform much better than
    # regression models


def q3_c_random_forest_valid(c_features_train, c_labels_train,
                             c_features_valid, c_labels_valid):
    """
    Takes the training and validation dataframes of features and labels
    (the classification version)
    Train a RandomForestClassifier model with n_estimators equals to
    100, 500, 1000, 1500, 2000
    Print the accuracy score of the model with different maximum depths
    using the validation datasets
    Note: I set the random state equals to 1, the max_depth equals to 6
    in order to make comparsions
    """
    # build model
    print('RandomForestClassifier')
    for n_estimators in [100, 500, 1000, 1500, 2000]:
        model = RandomForestClassifier(n_estimators=n_estimators,
                                       max_depth=6,
                                       random_state=1)
        model.fit(c_features_train, c_labels_train)
        # cauculate the stats
        train_acc = model.score(c_features_train, c_labels_train)
        test_acc = model.score(c_features_valid, c_labels_valid)
        print("N Estimators: %d  \t Train Accuracy:  %.4f  \
            Test Accuracy: %.4f" % (n_estimators, train_acc, test_acc))
    # We can see the n_estimators doesn't really affect the performance of
    # the model. Therefore, let's set the n_estimators into 1000


def q3_c_boost_valid(c_features_train, c_labels_train,
                     c_features_valid, c_labels_valid):
    """
    Takes the training and validation dataframes of features and labels
    (the classification version)
    Train a XGBClassifier model with learning_rate equals to
    0.01, 0.008, 0.005, 0.003, 0.001
    Print the accuracy score of the model with different maximum depths
    using the validation datasets
    Note: I set the random state equals to 1, the max_depth equals to 6,
    and the n_estimators equals to 1000 in order to make comparsions
    """
    # build model
    print('XGBClassifier')
    for learning_rate in [0.01, 0.008, 0.005, 0.003, 0.001]:
        model = XGBClassifier(n_estimators=1000, learning_rate=learning_rate,
                              max_depth=6, random_state=1,
                              eval_metric='mlogloss')
        model.fit(c_features_train, c_labels_train,
                  eval_set=[(c_features_valid, c_labels_valid)],
                  verbose=False)
        # cauculate the stats
        train_acc = model.score(c_features_train, c_labels_train)
        test_acc = model.score(c_features_valid, c_labels_valid)
        print("Learning Rate: %.3f  \t Train Accuracy:  %.4f  \
            Test Accuracy: %.4f" % (learning_rate, train_acc, test_acc))


def q3_c_plot():
    """
    Plots a chart with 3 tables as subplots showing the changing
    hyper-parameters, the training accuracy, and the testing accuracy
    of each generated machine learning model (classifier)
    Color the row with selected hyper-parameters in light green
    Titles the first table as "Validate Max Depth in
    DecisionTreeClassifier",
    the second table as "Validate N Estimators in RandomForestClassifier",
    and the third table as "Validate Learning Rate in XGBClassifier"
    Titles the chart as "Table Comparing the 3 Classification Models"
    Note: to save the plot generated from plotly, click the camera
    label on the top right of the popping website
    """
    select_color = 'lightgreen'
    other = 'lavender'
    fig = make_subplots(
        rows=3, cols=1,
        specs=[[{"type": "table"}],
               [{"type": "table"}],
               [{"type": "table"}]],
        subplot_titles=['Validate Max Depth in DecisionTreeClassifier',
                        'Validate N Estimators in RandomForestClassifier',
                        'Validate Learning Rate in XGBClassifier']
    )
    # DecisionTreeClassifier
    fig.add_trace(go.Table(
        header=dict(
            values=["Max Depth", "Training Accuracy",
                    "Testing Accuracy"],
            fill_color='paleturquoise',
            align="left"
        ),
        cells=dict(
            values=[[2, 5, 6, 7, 8],
                    [0.4409, 0.4809, 0.4949, 0.5049, 0.5203],
                    [0.4327, 0.4707, 0.4772, 0.4745, 0.4730]],
            fill_color=[[other, other, select_color, other, other]*5],
            align="left")
    ),
        row=1, col=1
    )
    # RandomForestClassifier
    fig.add_trace(go.Table(
        header=dict(
            values=["N Estimators", "Training Accuracy",
                    "Testing Accuracy"],
            fill_color='paleturquoise',
            align="left"
        ),
        cells=dict(
            values=[[100, 500, 1000, 1500, 2000],
                    [0.4953, 0.4949, 0.4952, 0.4956, 0.4961],
                    [0.4768, 0.4749, 0.4734, 0.4753, 0.4757]],
            fill_color=[[other, other, select_color, other, other]*5],
            align="left")
    ),
        row=2, col=1
    )
    # XGBClassifier
    fig.add_trace(go.Table(
        header=dict(
            values=["Learning Rate", "Training Accuracy",
                    "Testing Accuracy"],
            fill_color='paleturquoise',
            align="left"
        ),
        cells=dict(
            values=[[0.01, 0.008, 0.005, 0.003, 0.001],
                    [0.5667, 0.5562, 0.5417, 0.5299, 0.5141],
                    [0.4810, 0.4860, 0.4883, 0.4791, 0.4707]],
            fill_color=[[other, other, select_color, other, other]*5],
            align="left")
    ),
        row=3, col=1
    )

    fig.update_layout(
        title_text="Table Comparing the 3 Classifier Models",
    )
    fig.show()


def q3_c_final_model(c_features_train, c_labels_train, c_features_test,
                     c_labels_test, c_features_valid, c_labels_valid):
    """
    Takes the training, validation, and testing dataframes of features
    and labels (the classification version)
    Train a XGBClassifier model with the proper hyper-parameters I chose
    and fit the model with validation datasets as evalution for the model
    Print the training accuracy and testing accuracy of the model using
    the testing datasets
    Then,
    Plots a chart with 2 tables as subplots showing the training and testing
    accuracy of the final chosen XGBClassifier model and its actual range and
    prediction range in the first 30 rows of the testing data
    Titles the first table as "Testing Statistics of the XGBClassifier Model",
    and the second table as "Actual and Predicted Airbnb Range Comparison",
    Note: to save the plot generated from plotly, click the camera
    label on the top right of the popping website
    """
    print('Chosen XGBClassifier:')
    model = XGBClassifier(n_estimators=1000, learning_rate=0.005,
                          max_depth=6, eval_metric='mlogloss')
    model.fit(c_features_train, c_labels_train,
              eval_set=[(c_features_valid, c_labels_valid)],
              verbose=False)
    # cauculate the stats
    test_predictions = model.predict(c_features_test)
    train_acc = model.score(c_features_train, c_labels_train)
    test_acc = model.score(c_features_test, c_labels_test)
    print("Train Accuracy:  %.4f Test Accuracy: %.4f" % (train_acc, test_acc))

    error = pd.DataFrame({'actual': np.array(c_labels_test).flatten(),
                          'predict': test_predictions.flatten()})
    show = error.head(30)
    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{"type": "table"}, {"type": "table"}]],
        subplot_titles=['Testing Statistics of the XGBClassifier Model',
                        'Actual and Predicted Airbnb Range Comparison']
    )
    fig.add_trace(go.Table(
        header=dict(
            values=["Train Accuracy", "Test Accuracy"],
            fill_color='paleturquoise',
            align="left"
        ),
        cells=dict(
            values=[0.5417, 0.4856],
            fill_color='lavender',
            align="left")
    ),
        row=1, col=1
    )
    fig.add_trace(go.Table(
        header=dict(values=['Actual Range', 'Predicted Range'],
                    fill_color='paleturquoise',
                    align='left'),
        cells=dict(values=[show.actual, show.predict],
                   fill_color='lavender', align='left')
    ),
        row=1, col=2
    )
    fig.update_layout(
        title_text="Performance of the Selected XGBClassifier Model"
    )
    fig.show()


def q4_count(nyc_df, path, test=False):
    """
    Takes the merged New York City Airbnb dataframe, a path to the
    generated file, and a status for testing (set it to False as default)
    Plots a bar chart containing the top 10 most frequency word used as
    names of Airbnbs
    Titles the plot as "The Top 10 Most Frequent Words Used in names of
    New York Airbnbs"
    Saves the plot as the given path name
    """
    clean_name = nyc_df['name'].str.lower().str.cat(sep=' ')
    words = word_tokenize(clean_name)
    word_dist = FreqDist(words)
    if test is False:
        count_df = pd.DataFrame(word_dist.most_common(20),
                                columns=['Word', 'Frequency'])
        # print(count_df)
        # Here, I print the count_df to manually exclude meaningless words
        # such as "a" and "the"
        count_df = count_df.loc[count_df['Word'].isin(['room', 'bedroom',
                                                       'private',
                                                       'apartment',
                                                       'cozy', 'brooklyn',
                                                       'apt', 'east',
                                                       'spacious',
                                                       'studio']), :]
    else:
        count_df = pd.DataFrame(word_dist.most_common(10),
                                columns=['Word', 'Frequency'])

    sns.catplot(data=count_df, x='Word', y='Frequency', kind='bar',
                color='b')
    plt.xticks(rotation=-45)
    plt.title('The Top 10 Most Frequent Words Used in Names of '
              'New York Airbnbs')
    plt.savefig(path, bbox_inches='tight')


def main():
    # load data
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
    # Here, I check neighborhood names' differences in the airbnb_nyc dataset
    # and the average_price_nb_df before merging them together. I apply the
    # ckeck_diff() function from data_cleaning file here because I'd like to
    # munually fix some neighborhoods with name differences but actually
    # represent the same neighborhoods. However, according to the result,
    # both of the datasets are clean enough that I don't need to change names
    # manually.
    # name_difference = data_cleaning.check_diff(airbnb_nyc,
    #                                            average_price_nb_df)
    nyc_df = data_cleaning.merge_dataframe(airbnb_nyc, average_price_nb_df)
    geo_nyc_df = data_cleaning.to_geo_df(nyc_df)

    # question 1:
    geo_group_df = data_cleaning.group_nb_df(nyc_df,
                                             nyc_neighborhood_df)
    q1_mean_price(geo_group_df, nyc_neighborhood_df)
    q1_relative_price(geo_group_df, nyc_neighborhood_df)

    # question 2:
    geo_nyc_20 = geo_nyc_df.nlargest(20, 'number_of_reviews')
    q2_pop(nyc_neighborhood_df, geo_nyc_20)
    q2_ployly_table(geo_nyc_20)

    # question 3:
    filtered_data = nyc_df[['neighbourhood_group', 'room_type',
                            'minimum_nights', 'reviews_per_month',
                            'average_price', 'price']]
    filtered_data = filtered_data.dropna()
    # At first, I built several regression models and compared their stats
    # one-hot encode
    features = filtered_data.loc[:, filtered_data.columns != 'price']
    features = pd.get_dummies(features)
    labels = filtered_data['price']
    # split data into train 70%, validation 15%, and test 15%
    features_train, features_rem, labels_train, labels_rem = \
        train_test_split(features, labels, test_size=0.3, random_state=1)
    features_valid, features_test, labels_valid, labels_test = \
        train_test_split(features_rem, labels_rem, test_size=0.5,
                         random_state=1)

    q3_decision_tree_valid(features_train, labels_train, features_valid,
                           labels_valid)
    q3_random_forest_valid(features_train, labels_train, features_valid,
                           labels_valid)
    q3_boost_valid(features_train, labels_train, features_valid,
                   labels_valid)
    q3_plot()
    # Finally I choose the XGBRegressor with n_estimators equals 1500,
    # max_depth equals 5, and learning_rate equals 0.0013
    q3_final_model(features_train, labels_train, features_test, labels_test,
                   features_valid, labels_valid)

    q3_plot_price(filtered_data)
    # classifer:
    class_filtered_data = data_cleaning.c_filter_data(filtered_data)
    class_features = class_filtered_data.loc[:, class_filtered_data.
                                             columns != 'price_range']
    class_features = pd.get_dummies(class_features)
    class_labels = class_filtered_data['price_range']
    # split data into train 70%, validation 15%, and test 15%
    c_features_train, c_features_rem, c_labels_train, c_labels_rem = \
        train_test_split(class_features, class_labels, test_size=0.3,
                         random_state=1)
    c_features_valid, c_features_test, c_labels_valid, c_labels_test = \
        train_test_split(c_features_rem, c_labels_rem, test_size=0.5,
                         random_state=1)

    q3_c_decision_tree_valid(c_features_train, c_labels_train,
                             c_features_valid, c_labels_valid)
    q3_c_random_forest_valid(c_features_train, c_labels_train,
                             c_features_valid, c_labels_valid)
    q3_c_boost_valid(c_features_train, c_labels_train,
                     c_features_valid, c_labels_valid)

    q3_c_plot()
    q3_c_final_model(c_features_train, c_labels_train, c_features_test,
                     c_labels_test, c_features_valid, c_labels_valid)

    # question 4:
    q4_count(nyc_df, './q4_frq.png')


if __name__ == '__main__':
    main()
