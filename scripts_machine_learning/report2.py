import numpy as np
import pandas as pd
import geopandas as gpd
import datetime

from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler

# EXTRA
from statsmodels.stats.outliers_influence import variance_inflation_factor 
from sklearn.linear_model import LinearRegression

from scipy import stats as st
from scipy.stats import shapiro
from scipy.stats import normaltest
from scipy.stats import anderson

if __name__ == "__main__":
    # Importing data.
    file_name = "./data2.csv"
    data = pd.read_csv(file_name)
    data = data[data.columns[1:]]
    data["date"] = pd.to_datetime(data["date"])
    data.sort_values(by="date")

    # Creating train and test datasets.
    n_train = int(data.shape[0] / 3 * 2)

    features = data[data.columns[2:-1]]
    targets = data[data.columns[-1]]

    X_train = features.loc[:n_train, :]
    X_test = features.loc[n_train + 1:, :]

    y_train = targets.loc[:n_train]
    y_test = targets.loc[n_train + 1:]

    """
    table = []
    count = 0
    for feature in features.columns:
        count = count + 1
        row = ""
        
        row = row + str(count) + " & "
        row = row + feature + " & "
        row = row + "test  & "
        row = row + "test  & "

        row = row + str("{:.2f}".format(np.mean(features[feature]))) + " & "
        row = row + str("{:.2f}".format(st.mode(features[feature]).mode)) + " & "
        row = row + str("{:.2f}".format(np.var(features[feature]))) + " & "

        stat, p = shapiro(features[feature])
        row = row + str("{:.2f}".format(p))  + " & "

        stat, p = normaltest(features[feature])
        row = row + str("{:.2f}".format(p))  + " & "

        result = anderson(features[feature])

        stat = result.statistic
        cv = result.critical_values[2]

        if stat > cv:
            row = row + "$p > cv$\\" + "\\"
        else:
            row = row + "$p < cv$\\" + "\\"

        table.append(row)

    count = count + 1
    row = ""
    
    row = row + str(count) + " & "
    row = row + "Overdose" + " & "
    row = row + "test  & "
    row = row + "test  & "
    row = row + str("{:.2f}".format(np.mean(targets))) + " & "
    row = row + str("{:.2f}".format(st.mode(targets).mode)) + " & "
    row = row + str("{:.2f}".format(np.var(targets))) + " & "

    stat, p = shapiro(targets)
    row = row + str("{:.2f}".format(p))  + " & "

    stat, p = normaltest(targets)
    row = row + str("{:.2f}".format(p))  + " & "

    result = anderson(targets)

    stat = result.statistic
    cv = result.critical_values[2]

    if stat > cv:
        row = row + "$p > cv$\\" + "\\"
    else:
        row = row + "$p < cv$\\" + "\\"

    table.append(row)

    #for t in table:
    #    print(t)
    """

    ## CHECKING VIF (EXTRA) ###############
    X = features
    X = X.drop(labels=["tmean", "tmin", "EP_HBURD", "White"], axis="columns")

    scaler = MinMaxScaler().fit(X)
    X = pd.DataFrame(scaler.transform(X), columns=X.columns)

    features = X

    vif = pd.DataFrame()
    vif["feature"] = X.columns
    vif["VIF"] = [variance_inflation_factor(X.values, i) for i in range(len(X.columns))] 

    for i, row in vif.iterrows():
        print(row["feature"] + ": " + str(row["VIF"]))


    #######################################

    ## HERE ARE EXTRA ######################
    scaler = MinMaxScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    y_train = y_train.values
    y_train = y_train.reshape((-1, 1))

    y_test = y_test.values
    y_test = y_test.reshape((-1, 1))


    scaler = MinMaxScaler().fit(y_train)
    y_train = scaler.transform(y_train)
    y_test = scaler.transform(y_test)
    

    #######################################
    
    ## OLS (EXTRA) ########################
    print(X_train.shape)
    regressor = LinearRegression().fit(X_train, y_train)
    print(regressor.score(X_test, y_test))

    exit()

    print("Coeffs: ")
    i = 0
    for f in features.columns:
        print(f"{f}: {regressor.coef_[0, i]}")
        i = i + 1

    print("Intercept: ")
    print(regressor.intercept_)



    print("\ntest-set R^2: ", regressor.score(X_test, y_test))



    #######################################

    # Fitting a random forest model.
    regressor = RandomForestRegressor(n_estimators=10, random_state=0)
    regressor.fit(X_train, y_train)

    print("test-set R^2: ", regressor.score(X_test, y_test))



