import numpy as np
import pandas as pd
import geopandas as gpd
import datetime

if __name__ == "__main__":
    # Importing overdose data.
    file_name = "../data/overdose_deaths/overdose_deaths.csv"
    overdose_data = pd.read_csv(file_name)
    overdose_data_cols = overdose_data.columns[1:]
    overdose_data = overdose_data[overdose_data_cols]
    overdose_data["FIPS"] = overdose_data["FIPS"].apply(lambda x: str(x).zfill(5)).astype(object)

    # Adding population to compute deaths ratio.
    file_name = "../data_raw/demographic/csv/SVI_2020_US_county.csv"
    counties_pop = pd.read_csv(file_name)
    counties_pop = counties_pop[["FIPS", "E_TOTPOP"]]
    counties_pop["FIPS"] = counties_pop["FIPS"].apply(lambda x: str(x).zfill(5)).astype(object)

    total_deaths = overdose_data.merge(counties_pop, on="FIPS", how="left")
    cols = total_deaths.columns[1:-1]

    # Computing deaths ratios in 100,000 people.
    for c in cols:
        total_deaths[c] = total_deaths[c] / total_deaths["E_TOTPOP"] * 100000

    overdose_data = total_deaths[overdose_data.columns]

    # Importing precipitation data.
    file_name = "../data/precipitation/precipitation.csv"
    ppm_data = pd.read_csv(file_name)
    ppm_data_cols = ppm_data.columns[1:]
    ppm_data = ppm_data[ppm_data_cols]
    ppm_data = ppm_data.rename(columns={"GEOID":"FIPS"})
    ppm_data["FIPS"] = ppm_data["FIPS"].apply(lambda x: str(x).zfill(5)).astype(object)

    # Importing mean temperature data.
    file_name = "../data/tmean/tmean.csv"
    tmean_data = pd.read_csv(file_name)
    tmean_data_cols = tmean_data.columns[1:]
    tmean_data = tmean_data[tmean_data_cols]
    tmean_data = tmean_data.rename(columns={"GEOID":"FIPS"})
    tmean_data["FIPS"] = tmean_data["FIPS"].apply(lambda x: str(x).zfill(5)).astype(object)

    # Importing min temperature data.
    file_name = "../data/tmin/tmin.csv"
    tmin_data = pd.read_csv(file_name)
    tmin_data_cols = tmin_data.columns[1:]
    tmin_data = tmin_data[tmin_data_cols]
    tmin_data = tmin_data.rename(columns={"GEOID":"FIPS"})
    tmin_data["FIPS"] = tmin_data["FIPS"].apply(lambda x: str(x).zfill(5)).astype(object)

    # Importing max temperature data.
    file_name = "../data/tmax/tmax.csv"
    tmax_data = pd.read_csv(file_name)
    tmax_data_cols = tmax_data.columns[1:]
    tmax_data = tmax_data[tmax_data_cols]
    tmax_data = tmax_data.rename(columns={"GEOID":"FIPS"})
    tmax_data["FIPS"] = tmax_data["FIPS"].apply(lambda x: str(x).zfill(5)).astype(object)

    # Importing ethnicity data. X, Y of centroids are in this file.
    file_name = "../data_spatial/ethnicity/ethnicity.shp"
    eth_data = gpd.read_file(file_name)
    eth_data = eth_data[eth_data.columns[:-1]]
    eth_data = eth_data.rename(columns={"GEOID":"FIPS"})
    eth_data = eth_data.rename(columns={"P0020002":"Hispanic"})
    eth_data = eth_data.rename(columns={"P0020005":"White"})
    eth_data = eth_data.rename(columns={"P0020006":"Black"})
    eth_data = eth_data.rename(columns={"P0020008":"Asian"})
    eth_data["FIPS"] = eth_data["FIPS"].apply(lambda x: str(x).zfill(5)).astype(object)

    races = ["Hispanic", "White", "Black", "Asian"]
    for race in races:
        eth_data[race] = eth_data[race] / eth_data["P0010001"]

    # Importing demographic data.
    file_name = "../data/demographic/demographic.csv"
    demographic_data = pd.read_csv(file_name)
    demographic_data = demographic_data[demographic_data.columns[1:]]
    demographic_data["FIPS"] = demographic_data["FIPS"].apply(lambda x: str(x).zfill(5)).astype(object)

    # Joinin these datasets.
    data = overdose_data.merge(ppm_data, on="FIPS", how="left")
    data = data.merge(tmean_data, on="FIPS", how="left")
    data = data.merge(tmin_data, on="FIPS", how="left")
    data = data.merge(tmax_data, on="FIPS", how="left")
    data = data.merge(eth_data, on="FIPS", how="left")
    data = data.merge(demographic_data, on="FIPS", how="left")


    # Creating dataset.
    epocs = overdose_data.columns[1:]

    data_set = {"FIPS":           [],
                "date":           [],
                "year":           [],
                "season":         [],
                "month":          [],
                "ppm":            [],
                "tmin":           [],
                "tmean":          [],
                "tmax":           [],
                "Hispanic":       [],
                "White":          [],
                "Black":          [],
                "Asian":          [],
                "X":              [],
                "Y":              [],
                "EP_UNEMP":       [],
                "EP_HBURD":       [],
                "EP_NOHSDP":      [],
                "EP_UNINSUR":     [],
                "EP_CROWD":       [],
                "EP_NOVEH":       [],
                "overdose_rate":  []}

    for i, row in data.iterrows():
        for e in epocs:
            year = e.split("_")[0]
            month = e.split("_")[1].zfill(2)
            data_set["FIPS"].append(row["FIPS"])
            data_set["date"].append(datetime.datetime(year=int(year), month=int(month), day=1))

            if int(month) in [12, 1, 2]:
                season = 4
            elif int(month) in [3, 4, 5]:
                season = 1
            elif int(month) in [6, 7, 8]:
                season = 2
            else:
                season = 3
            data_set["year"].append(year)
            data_set["season"].append(season)
            data_set["month"].append(month)
            data_set["ppm"].append(row["ppm_" + year + "_" + month])
            data_set["tmin"].append(row["tmin" + year + "_" + month])
            data_set["tmean"].append(row["tmean" + year + "_" + month])
            data_set["tmax"].append(row["tmax" + year + "_" + month])
            data_set["Hispanic"].append(row["Hispanic"])
            data_set["White"].append(row["White"])
            data_set["Black"].append(row["Black"])
            data_set["Asian"].append(row["Asian"])
            data_set["X"].append(row["X"])
            data_set["Y"].append(row["Y"])
            data_set["EP_UNEMP"].append(row["EP_UNEMP"])
            data_set["EP_HBURD"].append(row["EP_HBURD"])
            data_set["EP_NOHSDP"].append(row["EP_NOHSDP"])
            data_set["EP_UNINSUR"].append(row["EP_UNINSUR"])
            data_set["EP_CROWD"].append(row["EP_CROWD"])
            data_set["EP_NOVEH"].append(row["EP_NOVEH"])
            
            data_set["overdose_rate"].append(row[e])

    data_set = pd.DataFrame(data_set)
    data_set = data_set.dropna()

    # Exporting this dataset.
    file_name = "data.csv"

    data_set.to_csv("data.csv")

    print("Data file exported to \"" + file_name + "\".")
