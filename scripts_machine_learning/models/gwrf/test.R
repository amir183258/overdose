library(dplyr)
library(SpatialML)

RDF <- random.test.data(60, 10, 3)
Coords <- RDF[,4:5]

grf <- grf(dep ~ X1 + X2, dframe=RDF, bw=10, kernel="adaptive", coords=Coords)

