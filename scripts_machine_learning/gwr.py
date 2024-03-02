import numpy as np
from mgwr.gwr import GWR, MGWR
from mgwr.sel_bw import Sel_BW
import geopandas as gp
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.preprocessing import StandardScaler 

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

    # Creating features (X) and targets (y) and coordinates.
    X = data[data.columns[:-1]]

    u = X["X"]
    v = X["Y"]

    X = X.drop(labels=["X", "Y"], axis="columns")

    y = data[data.columns[-1]]

    X = X.values
    y = y.values.reshape(-1, 1)

    scaler = StandardScaler().fit(X)
    X = scaler.transform(X)

    scaler = StandardScaler().fit(y)
    y = scaler.transform(y)

    u = u.values.reshape(-1, 1)
    u = u
    v = v.values.reshape(-1, 1)
    v = v

    coords = np.array(list(zip(u, v))).reshape(-1, 2)

    # GWR Model
    gwr_selector = Sel_BW(coords, y, X)
    gwr_bw = gwr_selector.search(bw_min=2)

    print("Selected Bandwidth: " + str(gwr_bw))

    model = GWR(coords, y, X, gwr_bw)
    gwr_results = model.fit()

    print(gwr_results.summary())
