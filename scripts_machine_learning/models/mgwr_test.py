import numpy as np
import pandas as pd

from mgtwr.sel import SearchMGWRParameter
from mgtwr.model import MGWR

from sklearn.preprocessing import StandardScaler 

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

    # Creating features (X) and targets (y) and coordinates.
    X = data[data.columns[:-1]]

    X = X.drop(labels=["X", "Y"], axis="columns")

    y = data[data.columns[-1]]
    y = y.values.reshape(-1, 1)

    # Standardizing the data
    scaler =  StandardScaler().fit(X.values)
    X = pd.DataFrame(scaler.transform(X.values), columns=X.columns)

    scaler =  StandardScaler().fit(y)
    y = scaler.transform(y)

    coords = data[["X", "Y"]]

    # MGWR Model
    sel_multi = SearchMGWRParameter(coords, X, y, kernel='gaussian', fixed=True, thread=5)
    bw = sel_multi.search(verbose=True, time_cost=True)

    print("**** Selected Bandwidth: " + str(bw) + "**** ")

    mgwr = MGWR(coords, X, y, bw, kernel='gaussian', fixed=True).fit()

    print(mgwr.R2)
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

    print("GWR done!!")
