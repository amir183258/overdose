import numpy as np
from mgwr.gwr import GWR, MGWR
from mgwr.sel_bw import Sel_BW
import geopandas as gp
import matplotlib.pyplot as plt
import pandas as pd


if __name__ == "__main__":
    # Importing data
    data_file = "data.csv"
    data = pd.read_csv(data_file)
    data = data[data.columns[1:]]

    ################## Here is for test. ########
    cols = data.columns
    data = data.values
    data = data[data[:, 0] == 2022]
    data = data[data[:, 2] == 1]

    data = pd.DataFrame(data, columns=cols)
    data = data[data.columns[3:]]

    ##############################################

    # Createing features (X) and targets (y).
    X = data[data.columns[:-1]]
    X_OLS = X

    u = X["X"]
    v = X["Y"]

    X = X.drop(labels=["X", "Y"], axis="columns")

    y = data[data.columns[-1]]

    ## Creating test and train datasets.
    n_train = int(X.values.shape[0] * 2 / 3)
    # X OLS
    X_OLS_train = X_OLS.values[:n_train, :]
    X_OLS_test = X_OLS.values[n_train:, :]

    # X GWR and MGWR
    #X_train = X.values[:n_train, :]
    X_train = X.values
    X_test = X.values[n_train:, :]

    # y and spatial components
    y = y.values.reshape(-1, 1)
    #y_train = y[:n_train, :]
    y_train = y
    y_test = y[n_train:, :]

    u = u.values.reshape(-1, 1)
    #u_train = u[:n_train, :]
    u_train = u
    u_test = u[n_train:, :]

    v = v.values.reshape(-1, 1)
    #v_train = v[:n_train, :]
    v_train = v
    v_test = v[n_train:, :]

    coords_train = np.array(list(zip(u_train, v_train))).reshape(-1, 2)

    coords_test = np.array(list(zip(u_test, v_test))).reshape(-1, 2)

    # OLS implementation.
    #scaler = MinMaxScaler().fit(X_OLS_train)
    #X_OLS_train = scaler.transform(X_OLS_train)
    #X_OLS_test = scaler.transform(X_OLS_test)

    #scaler = MinMaxScaler().fit(y_train)
    #y_train = scaler.transform(y_train)
    #y_test = scaler.transform(y_test)

    #regressor = LinearRegression().fit(X_OLS_train, y_train)

    #print("OLS R^2: ")
    #print(regressor.score(X_OLS_test, y_test))

    # GWR
    gwr_selector = Sel_BW(coords_train, y_train, X_train)
    gwr_bw = gwr_selector.search(bw_min=2)

    print("Selected Bandwidth: " + str(gwr_bw))

    model = GWR(coords_train, y_train, X_train, gwr_bw)
    gwr_results = model.fit()

    print(gwr_results.summary())

    exit()

    scale = gwr_results.scale
    residuals = gwr_results.resid_response

    pred_results = model.predict(coords_test, X_test, scale, residuals)

    print("GWR R^2: ")
    print(np.corrcoef(pred_results.predictions.flatten(), y_test.flatten())[0][1])
    


