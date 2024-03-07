import time
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler 

from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant

if __name__ == "__main__":
    # Importing data
    data_file = "./data/data.csv"
    data = pd.read_csv(data_file)
    data = data[data.columns[1:]]

    # Using 70% of data.
    train_index = int(len(data) * 70 / 100)

    data = pd.DataFrame(data.values[:train_index, :], columns=data.columns)

    # Standardizing the data.
    scaler =  StandardScaler().fit(data.values)
    data = pd.DataFrame(scaler.transform(data.values), columns=data.columns)

    # Creating features (X) and targets (y) and coordinates.
    X = data[data.columns[:-1]]
    y = data[data.columns[-1]]

    # OLS regression.
    start_time = time.time()
    regressor = OLS(y, add_constant(X)).fit()
    end_time = time.time()

    print(regressor.aic) 
    print(f"Modeling time: {end_time - start_time} seconds.")
    print("test-set R^2: ", regressor.rsquared)

    report = str(regressor.summary())

    with open("./ols/ols.txt", "w") as f:
        f.write(report)
    
    print("OLS is done!!")
