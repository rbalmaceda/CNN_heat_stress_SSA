options(java.parameters = "-Xmx32g")
library(loadeR)
library(transformeR)
library(downscaleR) # version v3.3.2
library(downscaleR.keras) # version devel
library(magrittr) # to operate with '%>%' or '%<>%'
library(here)
require(RColorBrewer)
require(visualizeR)
require(ggplot2)
require(gridExtra)

# Define custom_norm and normalizeGrid...
custom_norm <- function(array) {
  min_value <- min(array, na.rm = TRUE)
  max_value <- max(array, na.rm = TRUE)
  diff <- max_value - min_value
  
  out_array <- (array - min_value) / diff
}
normalizeGrid <- function(grid, time.frame = NULL) {
  if (is.null(time.frame)) {
    grid$Data <- custom_norm(grid$Data)
    
  } else if (time.frame == "daily") {
    grid$Data <- lapply(1:getShape(grid, "time"), FUN = function(day) {
      custom_norm(grid$Data[,day,,])
    }) %>% abind::abind(along = 0) %>% aperm(c(2,1,3,4))
    attr(grid$Data, "dimensions") <- c("var", "time", "lat", "lon")
  }
  
  return(grid)
}

###################################################
predictands=c('twMax','tmax')

var=predictands[2]

load("~/Documentos/Back_up_Sol/x_files.Rdata")
x=redim(x,drop=T)
x=subsetGrid(x,years=1991:2020,season=c(12,1,2))

if( var=="twMax"){
  load("~/Documentos/Back_up_Sol/y_twMax/y_files.Rdata") #Aca hay que cambiarlo
  y=subsetGrid(y,years=1991:2020,season=c(12,1,2))
  
} else {
  load("~/Documentos/Back_up_Sol/y_tmax/y_files.Rdata") #Aca hay que cambiarlo
  y=subsetGrid(y,years=1991:2020,season=c(12,1,2))
  
}

y$Dates$start=x$Dates[[1]]$start
y$Dates$end=x$Dates[[1]]$end

x=getTemporalIntersection(obs=y,prd=x,which.return = "prd")
y=getTemporalIntersection(obs=y,prd=x,which.return = "obs")

path='/home/usuario/Documentos/Back_up_Sol/Single/'



### Preparing the predictors######
x_scaled <- scaleGrid(x, type = "standardize")

name_sites_original <- y$Metadata$station_id
name_sites=c("87418","87345","87585","87480")

name_sites_original <- y$Metadata$name
name_sites=which(name_sites_original %in% c("MENDOZA-AERO","CORDOBA-OBSERVATORIO","BUENOS-AIRES", "ROSARIO-AERO" ))
name_sites=name_sites_original[name_sites]

sites=c(22,16,28,37)


### Loop over sites ######
# path="/home/usuario/Documentos/Back_up_Sol/h5_pred/"
for (site in sites) {
  y_site <- subsetGrid(y, station.id = y$Metadata$station_id[site])
  name_site <- y$Metadata$name[site]
  id_site <-  y$Metadata$station_id[site]
  xCoord <- y$xyCoords$x[site]
  yCoord <- y$xyCoords$y[site]
  print(paste(name_site, id_site))
  
  ### Loading cnn model...
  model <- load_model_hdf5(filepath = paste0(path,sprintf("cnn_stationID_%s%s_relu.h5", id_site, var)), 
                           custom_objects = c("custom_loss" = gaussianLoss(last.connection = "dense"))
  )
  
  ### Computing integrated gradients ------------
  saliency_grids <- integratedGradients(x = x_scaled,
                                        model = model,
                                        baseline = NULL,
                                        num_steps = 50, # 100
                                        model.info = list(first.connection            = "conv",
                                                          last.connection             = "dense",
                                                          channels                    = "last",
                                                          time.frames                 = NULL,
                                                          nature                      = NULL,
                                                          ind_TrainingPredictandSites = 1,
                                                          ind_TrainingPredictorSites  = NULL,
                                                          data.structure = NULL,
                                                          coords = y_site$xyCoords
                                        ),
                                        site = data.frame("x" = xCoord, "y" = yCoord),
                                        batch = 200) # 80
  
  ### Saving integrated gradients ------------
  saveRDS(object = saliency_grids, 
          file = sprintf("saliency_%s%s_numsteps_training.rds", var, name_site))
  
  ### Free memory ------------
  k_clear_session()
  model <- saliency_grids <- NULL
  gc()
}

######## ploteo ########
cb <- brewer.pal(n = 9, "YlGn")
cb <- c("#FFFFFF", cb)
at <- c(0, seq(0.01, 0.9, length.out = 9)) 
predictors=c('termo','wind','high-levels')

# sites=sites[-1]
for (site in sites) {
  name_site <- y$Metadata$name[site]
  id_site <-  y$Metadata$station_id[site]
  y_site <- subsetGrid(y, station.id = id_site)
  print(paste(name_site, id_site))
  
  # Load saliency maps...
  saliency_grids <- readRDS(sprintf("saliency_%s%s_numsteps_training.rds",var, name_site))
  saliency_grids$Data %<>% abs() 
  filter <- 0.0015
  saliency_grids$Data[saliency_grids$Data < filter] <- 0
  for (i in 1:dim(saliency_grids$Data)[1]) {
    saliency_grids$Data[i,,,] <- saliency_grids$Data[i,,,] / sum(saliency_grids$Data[i,,,]) 
  }
  
  saliency_grids_abs_mean <- climatology(saliency_grids, 
                                         clim.fun = list(FUN = "mean"))
  saliency_grids_abs_mean_normalized <- normalizeGrid(saliency_grids_abs_mean)
  rm(saliency_grids_abs_mean)

  ## ---- Variable selection ----
for (j in 1:2){
  predictor=predictors[j]
  
  if( predictor=='termo'){
  saliency_grids_abs_mean_normalized2=subsetGrid(saliency_grids_abs_mean_normalized,var = c("q@700","q@1000","q@850","t@1000","t@850","t@700"))
  }
  if(predictor=='wind'){
  saliency_grids_abs_mean_normalized2=subsetGrid(saliency_grids_abs_mean_normalized,var = c("v@700","v@1000","v@850","u@1000","u@850","u@700"))
  }
  if(predictor=='high-levels'){
    saliency_grids_abs_mean_normalized2=subsetGrid(saliency_grids_abs_mean_normalized,var = c("t@250","t@500","v@700","q@250","q@500","z@500"))
    
  }
  # saliency_grids_abs_mean%<>% gridArithmetics(100, operator = "*")
  # Display the saliency maps for 
  # the selected site....
  p= spatialPlot(saliency_grids_abs_mean_normalized2, 
                 backdrop.theme = "countries",
                 # xlim = c(-73,-60), ylim = c(-40,-20),#layout=c(8,1),as.table = TRUE,
                 col.regions = cb,
                 at = at,
                 colorkey = TRUE,
                 set.min = at[1], set.max = at[length(at)],
                 sp.layout = list(list(sp::SpatialPoints(y_site$xyCoords),
                                       first = FALSE,
                                       col = "black",
                                       pch = 16)
                 )
  )
  ggsave(plot=grid.arrange(p),filename = sprintf("saliency_%s%s%s%s.png",var, name_site,'_training_',predictor),device="png",dpi=300,width=5,height =5)
  
  # Free memory...
  gc()
}
}

############################ 2024############################
summer='23_24'
if (summer=="23_24"){
assign(value=get( load("~/Documentos/Back_up_Sol/x_2024_files.Rdata")),x = "x_s")
x_2024=subsetGrid(x_s, season = c(12,1,2))
x_2024 <- scaleGrid(grid = x_2024, base = subsetGrid(x, season = c(12,1,2)), type = "standardize")  #estandarizo usando los datos del periodo de TRAIN
x_pred=x_2024
}
if(summer=='22'){
x_pred=x_2022
}

for (site in sites) {
  y_site <- subsetGrid(y, station.id = y$Metadata$station_id[site])
  name_site <- y$Metadata$name[site]
  id_site <-  y$Metadata$station_id[site]
  xCoord <- y$xyCoords$x[site]
  yCoord <- y$xyCoords$y[site]
  print(paste(name_site, id_site))
  
### Loading cnn model...
path="/home/usuario/Documentos/Back_up_Sol/h5_pred/"
model <- load_model_hdf5(filepath = paste0(path,sprintf("cnn_stationID_%s%s_relu_23_24.h5", id_site, var)), 
                         custom_objects = c("custom_loss" = gaussianLoss(last.connection = "dense"))
)

### Computing integrated gradients ------------
saliency_grids <- integratedGradients(x = x_pred,
                                      model = model,
                                      baseline = NULL,
                                      num_steps = 50, # 100
                                      model.info = list(first.connection            = "conv",
                                                        last.connection             = "dense",
                                                        channels                    = "last",
                                                        time.frames                 = NULL,
                                                        nature                      = NULL,
                                                        ind_TrainingPredictandSites = 1,
                                                        ind_TrainingPredictorSites  = NULL,
                                                        data.structure = NULL,
                                                        coords = y_site$xyCoords
                                      ),
                                      site = data.frame("x" = xCoord, "y" = yCoord),
                                      batch = NULL) # 80
### Saving integrated gradients ------------
# saveRDS(object = saliency_grids, 
#         file = sprintf("saliency_%s%s_numsteps_2024.rds", var, name_site))

saveRDS(object = saliency_grids, 
        file = sprintf("saliency_%s%s%s_numsteps.rds", var, name_site,summer))

### Free memory ------------
k_clear_session()
model <- saliency_grids <- NULL
gc()
}

######################## ploteo#################################

cb <- brewer.pal(n = 9, "YlGn")
cb <- c("#FFFFFF", cb)
at <- c(0, seq(0.01, 0.9, length.out = 9)) 
predictors=c('termo','wind','high-levels')

# sites=sites[-1]
for (site in sites) {
  name_site <- y$Metadata$name[site]
  id_site <-  y$Metadata$station_id[site]
  y_site <- subsetGrid(y, station.id = id_site)
  print(paste(name_site, id_site))
  
  # Load saliency maps...
  saliency_grids <- readRDS(sprintf("saliency_%s%s%s_numsteps.rds", var, name_site,summer))
  saliency_grids$Data %<>% abs() 
  filter <- 0.0015
  saliency_grids$Data[saliency_grids$Data < filter] <- 0
  for (i in 1:dim(saliency_grids$Data)[1]) {
    saliency_grids$Data[i,,,] <- saliency_grids$Data[i,,,] / sum(saliency_grids$Data[i,,,]) 
  }
  # which(saliency_grids$Dates$start=="2024-01-21 23:00:00 GMT")
  summ=subsetDimension(saliency_grids,dimension = 'time',indices = 22:74)
  saliency_grids_abs_mean <- climatology(summ, 
                                         clim.fun = list(FUN = "mean"))
  
  saliency_grids_abs_mean_normalized <- normalizeGrid(saliency_grids_abs_mean)
  
  ## ---- Variable selection ----
  for (j in 1:2){
    predictor=predictors[j]
    
    if( predictor=='termo'){
      saliency_grids_abs_mean_normalized2=subsetGrid(saliency_grids_abs_mean_normalized,var = c("q@700","q@1000","q@850","t@1000","t@850","t@700"))
    }
    if(predictor=='wind'){
      saliency_grids_abs_mean_normalized2=subsetGrid(saliency_grids_abs_mean_normalized,var = c("v@700","v@1000","v@850","u@1000","u@850","u@700"))
    }
    if(predictor=='high-levels'){
      saliency_grids_abs_mean_normalized2=subsetGrid(saliency_grids_abs_mean_normalized,var = c("t@250","t@500","v@700","q@250","q@500","z@500"))
      
    }
  p= spatialPlot(saliency_grids_abs_mean_normalized2, 
                 backdrop.theme = "countries",
                 # xlim = c(-73,-60), ylim = c(-40,-20),#layout=c(8,1),as.table = TRUE,
                 col.regions = cb,
                 at = at,
                 colorkey = TRUE,
                 set.min = at[1], set.max = at[length(at)],
                 sp.layout = list(list(sp::SpatialPoints(y_site$xyCoords),
                                       first = FALSE,
                                       col = "black",
                                       pch = 16)
                 )
  )# %>% print()
  #  dev.off()
  ggsave(plot=grid.arrange(p),filename = sprintf("saliency_%s%s%s%s_summer.png",var, name_site,summer,predictor),device="png",dpi=300,width=5,height =5)
  
  # Free memory...
  gc()
}

}

######## 2022#######
saliency_grids <- readRDS(sprintf("saliency_%s%s_numsteps_2022.rds",var, name_site))


x_2022=subsetGrid(x_s, season = c(12,1,2),years = c(2021,2022))
x_2022 <- scaleGrid(grid = x_2022, base = subsetGrid(x, season = c(12,1,2)), type = "standardize")  #estandarizo usando los datos del periodo de TRAIN

x_2023=subsetGrid(x_s, season = c(12,1,2),years = c(2022,2023))
x_2023 <- scaleGrid(grid = x_2023, base = subsetGrid(x, season = c(12,1,2)), type = "standardize")  #estandarizo usando los datos del periodo de TRAIN
