import numpy as np
import pandas as pd

if __name__ == "__main__":
    data = pd.read_csv("./data.csv")
    data = data[data.columns[1:]]

    data_index = int(len(data) * 30 / 100)

    data = pd.DataFrame(data.values[data_index:, :], columns=data.columns)

    data.to_csv("./data.csv")

    print("Done! Data are ready for models.")
