import time
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler 
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import cross_val_score

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
    n_est = list(range(700, 1001, 100))
    max_features = [x / 10 for x in range(1, 11, 1)] 

    max_r2 = 0
    best_n = 0
    best_mf = 0
    for n in n_est:
        for mf in max_features:
            regressor = RandomForestRegressor(n_estimators=n, max_features=mf)

            scores = cross_val_score(regressor, X, y, cv=10)
            score = scores.mean()

            print(f"n_estimators = {n}, max_features = {mf}: R^2 = {score}")

            if score > max_r2:
                max_r2 = score
                best_n = n
                best_mf = mf
    print(f"Best n_estimators: {best_n}, Best max_features: {best_mf}")

    params_rf = {"n_estimators": best_n, "max_features": best_mf}

    # RandomForest regression.
    start_time = time.time()
    regressor = RandomForestRegressor(params_rf).fit(X, y)

    #grid_search = GridSearchCV(RandomForestRegressor(), param_grid, cv=10)
    #grid_search.fit(X, y)

    end_time = time.time()

    print("Time:", end_time - start_time)

    print("test-set R^2: ", regressor.score(X_test, y_test))
    #print(grid_search.best_params_)

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
