import time
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler 
from sklearn.linear_model import LinearRegression

if __name__ == "__main__":
    # Importing data
    data_file = "./data/data.csv"
    data = pd.read_csv(data_file)
    data = data[data.columns[1:]]

    # Createing test and train data.
    train_index = int(len(data) * 70 / 100)

    train_data = data.values[:train_index, :]
    test_data = data.values[train_index:, :]

    train_data = pd.DataFrame(train_data, columns=data.columns)
    test_data = pd.DataFrame(test_data, columns=data.columns)

    # Standardizing the data.
    scaler = StandardScaler().fit(train_data)

    train_data = pd.DataFrame(scaler.transform(train_data), columns=data.columns)
    test_data = pd.DataFrame(scaler.transform(test_data), columns=data.columns)

    # Creating features (X) and targets (y) and coordinates.
    X = train_data[data.columns[:-1]]
    y = train_data[data.columns[-1]]

    X_test = test_data[data.columns[:-1]]
    y_test = test_data[data.columns[-1]]

    # OLS regression.
    start_time = time.time()
    regressor = LinearRegression().fit(X, y)
    end_time = time.time()

    print("Train data R^2: ", regressor.score(X,y))
    print("Test data R^2: ", regressor.score(X_test, y_test))

    exit()
    print(regressor.aic) 
    print(f"Modeling time: {end_time - start_time} seconds.")
    print("test-set R^2: ", regressor.rsquared)

    report = str(regressor.summary())

    with open("./ols/ols.txt", "w") as f:
        f.write(report)
    
    print("OLS is done!!")
