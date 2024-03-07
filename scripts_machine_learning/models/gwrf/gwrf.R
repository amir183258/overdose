library(dplyr)
library(SpatialML)

# Reading the data.
dataFolder <- "./data/"
train_data <- read.csv(paste0(dataFolder, "train_data.csv"))
test_data <- read.csv(paste0(dataFolder, "test_data.csv"))

Coords <- train_data[c("X", "Y")]

# Computing the best bandwidth.
#bw <- grf.bw(overdose_rate ~ year + season + month + ppm + tmean + Hispanic + Black + Asian + EP_UNEMP + EP_HBURD + EP_NOHSDP + EP_UNINSUR + EP_CROWD + EP_NOVEH,
#	     dataset=train_data,
#	     kernel="adaptive",
#	     coords=Coords,
#	     nthreads=5)
#bw <- bw$best.bw

#print(bw)

#quit(save="no")

grf.model <- grf(overdose_rate ~ year + season + month + ppm + tmean + Hispanic + Black + Asian + EP_UNEMP + EP_HBURD + EP_NOHSDP + EP_UNINSUR + EP_CROWD + EP_NOVEH,
		 dframe=train_data,
		 bw=40,
		 ntree=10,
		 kernel="adaptive",
		 mtry=4,
		 coords=Coords, geo.weighted=TRUE)

# Results
write.csv(grf.model$Local.Variable.Importance, "./gwrf_results/local_importance.csv")

write.csv(grf.model$LGofFit, "./gwrf_results/goodnes_fit.csv")

y_predict = predict.grf(grf.model, test_data, x.var.name="X", y.var.name="Y", local.w=1, global.w=0)
write.csv(y_predict, "./gwrf_results/predict.csv")
