import pandas as pd
import numpy as np

if __name__ == "__main__":
    actual = pd.read_csv("./data/test_data.csv")["overdose_rate"].values
    predict = pd.read_csv("./predict.csv")["x"].values

    corr_matrix = np.corrcoef(actual, predict)
    corr = corr_matrix[0,1]
    R_sq = corr**2
     
    print(R_sq)


