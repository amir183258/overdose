import time
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler 
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import GridSearchCV

if __name__ == "__main__":
    # Importing data
    data_file = "./data/data.csv"
    data = pd.read_csv(data_file)
    data = data[data.columns[1:]]

    # Creating test and train data.
    train_index = int(len(data) * 70 / 100)

    train_data = data.values[:train_index, :]
    test_data = data.values [train_index:, :]

    # Standardizing the data.
    scaler =  StandardScaler().fit(train_data)
    train_data = pd.DataFrame(scaler.transform(train_data), columns=data.columns)
    test_data = pd.DataFrame(scaler.transform(test_data), columns=data.columns)

    # Creating features (X) and targets (y) and coordinates.
    X = train_data[data.columns[:-1]]
    y = train_data[data.columns[-1]]

    X_test = test_data[data.columns[:-1]]
    y_test = test_data[data.columns[-1]]

    # Parameters for grid search.
    param_grid = {"n_estimators": [200, 500, 1000], "max_features": [1, 4, 8]}

    # RandomForest regression.
    start_time = time.time()
    #regressor = RandomForestRegressor(n_estimators=100).fit(X, y)

    grid_search = GridSearchCV(RandomForestRegressor(), param_grid, cv=10)
    grid_search.fit(X, y)

    end_time = time.time()

    print("Time:", end_time - start_time)

    print("test-set R^2: ", grid_search.score(X_test, y_test))
    print(grid_search.best_params_)

    #print("test-set R^2: ", regressor.score(X, y))
    #print("test-set R^2: ", regressor.score(X_test, y_test))

    """

    print(regressor.aic) 
    print(f"Modeling time: {end_time - start_time} seconds.")
    print("test-set R^2: ", regressor.rsquared)

    report = str(regressor.summary())

    with open("./ols/ols.txt", "w") as f:
        f.write(report)
    
    print("OLS is done!!")
    """
