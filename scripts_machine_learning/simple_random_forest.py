import numpy as np
import pandas as pd
import geopandas as gpd
import datetime

from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler

# EXTRA
from statsmodels.stats.outliers_influence import variance_inflation_factor 
from sklearn.linear_model import LinearRegression

if __name__ == "__main__":
    # Importing data.
    file_name = "./data.csv"
    data = pd.read_csv(file_name)
    data = data[data.columns[1:]]
    data["date"] = pd.to_datetime(data["date"])
    data.sort_values(by="date")

    # Creating train and test datasets.
    n_train = int(data.shape[0] / 3 * 2)

    features = data[data.columns[2:-1]]
    targets = data[data.columns[-1]]

    X_train = features.loc[:n_train, :]
    X_test = features.loc[n_train + 1:, :]

    y_train = targets.loc[:n_train]
    y_test = targets.loc[n_train + 1:]

    ## CHECKING VIF (EXTRA) ###############
    X = features
    vif = pd.DataFrame()
    vif["feature"] = X.columns
    vif["VIF"] = [variance_inflation_factor(X.values, i) for i in range(len(X.columns))] 

    print(vif)
    exit()

    #######################################

    ## HERE ARE EXTRA ######################
    scaler = MinMaxScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)


    #######################################
    
    ## OLS (EXTRA) ########################
    regressor = LinearRegression().fit(X_train, y_train)
    print("test-set R^2: ", regressor.score(X_test, y_test))

    #######################################

    # Fitting a random forest model.
    regressor = RandomForestRegressor(n_estimators=100, random_state=0)
    regressor.fit(X_train, y_train)

    print("test-set R^2: ", regressor.score(X_test, y_test))



