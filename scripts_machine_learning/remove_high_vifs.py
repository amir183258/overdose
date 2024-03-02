import numpy as np
import pandas as pd
import geopandas as gpd
import datetime

from sklearn.preprocessing import StandardScaler, MinMaxScaler

# EXTRA
from statsmodels.stats.outliers_influence import variance_inflation_factor 


if __name__ == "__main__":
    # Importing data.
    file_name = "./data.csv"
    data = pd.read_csv(file_name)
    data = data[data.columns[1:]]
    data["date"] = pd.to_datetime(data["date"])
    data.sort_values(by="date")

    features = data[data.columns[2:-1]]
    targets = data[data.columns[-1]]

    # Examining the data for best vifs combination.
    X = features
    X = X.drop(labels=["tmin", "tmax", "White"], axis="columns")
    features = features.drop(labels=["tmin", "tmax", "White"], axis="columns")

    scaler = StandardScaler().fit(X)
    X = pd.DataFrame(scaler.transform(X), columns=X.columns)

    vif = pd.DataFrame()
    vif["feature"] = X.columns
    vif["VIF"] = [variance_inflation_factor(X.values, i) for i in range(len(X.columns))] 

    for i, row in vif.iterrows():
        print(row["feature"] + ": " + str(row["VIF"]))
    print("")

    # Creating final data.
    final_data = features
    final_data["overdose_rate"] = targets

    # Exporting the data.
    file_name = "data.csv"

    final_data.to_csv("data.csv")

    print("New Data file exported to \"" + file_name + "\".")
