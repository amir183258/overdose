library(dplyr)
library(SpatialML)

# Reading the data.
dataFolder <- "./data/"
train_data <- read.csv(paste0(dataFolder, "train_data.csv"))
test_data <- read.csv(paste0(dataFolder, "test_data.csv"))

print("hi")

Coords <- train_data[c("X", "Y")]

grf.model <- grf(overdose_rate ~ year + season + month + ppm + tmean + Hispanic + Black + Asian + EP_UNEMP + EP_HBURD + EP_NOHSDP + EP_UNINSUR + EP_CROWD + EP_NOVEH,
		 dframe=train_data,
		 bw=400,
		 ntree=10,
		 kernel="adaptive",
		 mtry=4,
		 coords=Coords, geo.weighted=TRUE)

y_predict = predict.grf(grf.model, test_data, x.var.name="X", y.var.name="Y", local.w=1, global.w=0)
write.csv(y_predict, "predict.csv")

