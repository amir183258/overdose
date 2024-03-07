import time
import numpy as np
import pandas as pd

import libpysal as ps
from mgwr.gwr import GWR
from mgwr.sel_bw import Sel_BW

from sklearn.preprocessing import StandardScaler 
from sklearn.metrics import r2_score

if __name__ == "__main__":
    # Importing data
    data_file = "./data/data.csv"
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

    # Train and Test data.
    train_index = int(len(data) * 70 / 100)

    train_data = pd.DataFrame(data.values[:train_index, :], columns=data.columns)
    test_data = pd.DataFrame(data.values[train_index:, :], columns=data.columns)

    # Coordinates
    coords = train_data[["X", "Y"]].values
    coords_test = test_data[["X", "Y"]].values

    # Standardizing the data
    scaler =  StandardScaler().fit(train_data)
    train_data = pd.DataFrame(scaler.transform(train_data), columns=data.columns)
    test_data = pd.DataFrame(scaler.transform(test_data), columns=data.columns)

    # Creating features (X) and targets (y) and coordinates.
    X = train_data[data.columns[:-1]]
    X_test = test_data[data.columns[:-1]]

    X = X.drop(labels=["X", "Y"], axis="columns")
    X_cols = X.columns
    X = X.values
    X_test = X_test.drop(labels=["X", "Y"], axis="columns")
    X_test = X_test.values

    y = train_data[data.columns[-1]]
    y = y.values.reshape(-1, 1)
    y_test = test_data[data.columns[-1]]
    y_test = y_test.values.reshape(-1, 1)

    # GWR Model
    #gwr_selector = Sel_BW(coords, y, X, kernel="gaussian", spherical=True)
    #bw = gwr_selector.search()

    bw = 71.0

    print("**** Selected Bandwidth: " + str(bw) + "**** ")

    start_time = time.time()
    model = GWR(coords, y, X, bw, kernel="gaussian", spherical=True)
    gwr_results = model.fit()
    end_time = time.time()

    print("**** Mode done after %s seconds. ****" % (end_time - start_time))

    gwr_results.summary()

    cols = ["Intercept"]
    for c in X_cols:
        cols.append(c)

    parameters = pd.DataFrame(gwr_results.params, columns=cols)
    parameters.to_csv("./gwr/gwr_coeffs.csv")

    tvalues = pd.DataFrame(gwr_results.tvalues, columns=cols)
    tvalues.to_csv("./gwr/gwr_tvalues.csv")

    residuals = pd.DataFrame(gwr_results.resid_response)
    residuals.to_csv("./gwr/gwr_residuals.csv")

    localR2 = pd.DataFrame(gwr_results.localR2)
    localR2.to_csv("./gwr/local_r2.csv")

    scale = gwr_results.scale
    residuals = gwr_results.resid_response
    pred_results = model.predict(coords_test, X_test, scale, residuals)

    print("Test data-set R^2:", np.corrcoef(pred_results.predictions.flatten(), y_test.flatten())[0][1])


    exit()

    # Exporting the report
    report_name = "gwr_report.txt"

    model_name = "GWR"

    data_shape_r = data.shape[0]
    data_shape_c = data.shape[1]

    features = [d for d in data.columns[:-1]]
    target = data.columns[-1]

    crs = "WGS84 / Decimal Degree"
    bandwidth = bw
    
    degree_of_freedom = gwr.df_model
    df_residuals = gwr.df_reside

    rss= gwr.RSS
    r2 = gwr.R2

    adj_r2 = gwr.adj_R2

    aic = gwr.aic
    aicc = gwr.aicc

    tvalues = gwr.tvalues

    betas = gwr.betas
    residuals = gwr.reside

    report = f"Model:\t{model_name}\n"
    report = report + f"Data Shape:\t\t{data_shape_r} x {data_shape_c}\n"
    report = report + f"Features:\n{features}\n\n"
    report = report + f"CRS:\t\t{crs}\n"
    report = report + f"Bandwidth:\t\t{bw}\n"
    report = report + f"Degree of Freedom:\t\t{degree_of_freedom}\n"
    report = report + f"RSS:\t\t{rss}\n"
    report = report + f"R^2:\t\t{r2}\n"
    report = report + f"Adjusted R^2:\t\t{adj_r2}\n"
    report = report + f"AIC:\t\t{aic}\n"
    report = report + f"AICc:\t\t{aicc}\n"

    with open("./gwr/" + report_name, "w") as f:
        f.write(report)

    cols = ["Intercept"]
    for c in X.columns:
        cols.append(c)

    betas = pd.DataFrame(betas, columns=cols)
    betas.to_csv("./gwr/gwr_coeffs.csv")

    residuals = pd.DataFrame(residuals)
    residuals.to_csv("./gwr/residuals.csv")

    print("GWR done!!")
