import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler

if __name__ == "__main__":
    data = pd.read_csv("../data/data.csv")
    data = data[data.columns[1:]]

    # Here ARE TEST ####################
    cols = data.columns
    row_numbers =  len(data)

    data = data.values[:int(row_numbers * 0.003), :]
    #data = data.values[:5000, :]

    data = pd.DataFrame(data, columns=cols)
    ####################################

    train_index = int(len(data) * 70 / 100)

    train_data = pd.DataFrame(data.values[:train_index, :], columns=data.columns)
    test_data = pd.DataFrame(data.values[train_index:, :], columns=data.columns)

    # Standardizing the data.
    scaler = StandardScaler().fit(train_data)

    X_train = train_data["X"]
    Y_train = train_data["Y"]

    train_data = pd.DataFrame(scaler.transform(train_data), columns=data.columns)
    train_data["X"] = X_train
    train_data["Y"] = Y_train

    X_test = test_data["X"]
    Y_test = test_data["Y"]

    test_data = pd.DataFrame(scaler.transform(test_data), columns=data.columns)
    test_data["X"] = X_test
    test_data["Y"] = Y_test

    train_data.to_csv("./data/train_data.csv", index=False)
    test_data.to_csv("./data/test_data.csv", index=False)

    print("Done!!")
