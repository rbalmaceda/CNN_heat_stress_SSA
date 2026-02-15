get_integrated_gradients <- function(input, model, baseline = NULL, num_steps = 50, site) {
  # If baseline is not provided, start with a zero field
  # having same dimensions as the input field.
  if (is.null(baseline)) {
    predictor_dims <- dim(input)
    baseline <- baseline_array <- array(0, dim = predictor_dims)
  }
  
  
  # 1. Do interpolation.
  interpolated_input <- list()
  for (step in 1:num_steps) interpolated_input[[step]] <- baseline + (step / num_steps) * (input - baseline)
  
  
  # 2. Get the gradients
  grads = list()
  for (index_int_input in 1:length(interpolated_input))  {
    grads[[index_int_input]] <- get_gradients(input = interpolated_input[[index_int_input]], 
                                              site = site,
                                              model = model)
  }
  grads <- tf$convert_to_tensor(grads, dtype = tf$float32)
  
  # 3. Approximate the integral using the trapezoidal rule
  grads = (grads[1:(num_steps-1),,,,] + grads[2:num_steps,,,,]) / 2.0  ### generalizar esto para inputs que no sean 4D!!!
  avg_grads = tf$reduce_mean(grads, axis = 0L)
  
  # 4. Calculate integrated gradients and return
  integrated_grads <- (input - baseline) * avg_grads
  return(integrated_grads %>% as.array())
}


get_gradients <- function(input, site, model) {
  input <- tf$cast(input, tf$float32)
  
  with(tf$GradientTape() %as% tape, {
    tape$watch(input)
    preds <- model(input)[,site]
  })
  
  grads <- tape$gradient(preds, input)
}


integratedGradients <- function(x = x,
                                model = model,
                                baseline = NULL,
                                num_steps = 50,
                                model.info = list(first.connection            = "conv",
                                                  last.connection             = "dense",
                                                  channels                    = "last",
                                                  time.frames                 = NULL,
                                                  nature                      = NULL,
                                                  ind_TrainingPredictandSites = NULL,
                                                  ind_TrainingPredictorSites  = NULL,
                                                  data.structure = NULL,
                                                  coords = NULL
                                ),
                                site = NULL,
                                saliency.fun = NULL,
                                batch = NULL) {
  
  ### Eliminate the 'member' dimension
  if (getShape(x, "member") > 1) {
    stop("No multi-member grids are allowed. Please consider using subsetGrid.")
  } else {
    x %<>% redim(drop = TRUE, member = FALSE)
  }
  
  ### Prepare input 'x' data based on `model.info` arguments
  data.structure <- list()
  data.structure$x.global <- list()
  attr(data.structure, "first.connection") <- model.info[["first.connection"]]
  attr(data.structure, "last.connection") <- model.info[["last.connection"]]
  attr(data.structure,"time.frames") <- model.info[["time.frames"]]
  attr(data.structure, "channels") <- model.info[["channels"]]
  attr(data.structure, "first.connection") <- model.info[["first.connection"]]
  attr(data.structure, "nature") <- model.info[["nature"]]
  attr(data.structure, "indices_noNA_y") <- model.info[["ind_TrainingPredictandSites"]]
  attr(data.structure, "indices_noNA_x") <- model.info[["ind_TrainingPredictorSites"]]
  attr(data.structure$x.global,"data.structure") <- model.info[["data.structure"]]
  
  if (model.info[["first.connection"]] == "conv") {
    x_input <- prepareNewData.keras(x, data.structure = data.structure)$x.global$member_1
  } else {
    ### to do: for local and spatial predictors in dense networks...
    
  }
  
  ### Match desired 'x,y' coordinated with output neuron
  output_neuron <- if (is.data.frame(site)) { 
    
    if (is.data.frame(model.info[["coords"]])) {### for irregular predictand grids
      coords_trainingSites <- model.info$coords[model.info[["ind_TrainingPredictandSites"]], ]
      ind_x <- which(site$x == coords_trainingSites$x)
      ind_y <- which(site$y == coords_trainingSites$y)
      intersect(ind_x, ind_y)
      
    } else if (is.list(model.info[["coords"]])) { ### for regular predictand grids
      ind_coords_x <- which(site$x == model.info[["coords"]]$x)
      ind_coords_y <- which(site$y == model.info[["coords"]]$y)
      aux_mat <- array(FALSE, dim = c(length(model.info[["coords"]]$x), length(model.info[["coords"]]$y)))
      aux_mat[ind_coords_x, ind_coords_y] <- TRUE
      aux_vector <- as.vector(aux_mat)[model.info[["ind_TrainingPredictandSites"]]]
      out_neuron <- which(aux_vector)
      if (length(out_neuron) == 0) {
        stop("The selected site was not optimized during the training phase and 
            therefore is not represented by any output neuron of the neural
            network.") 
      } else {
        out_neuron 
      }
    }  
    
  } else {
    site
  }
  
  ### Compute the integrated gradients
  array_int_grads <- if (is.null(batch)) {
    get_integrated_gradients(input = x_input, 
                             site = output_neuron, 
                             model = model,
                             baseline = baseline, 
                             num_steps = num_steps) 
  } else {
    samples <- dim(x_input)[1]
    init_batches <- seq(1, samples, batch)
    end_batches <- c(seq(batch, samples, batch), samples) %>% unique()
    mapply(init_batches, end_batches, FUN = function(init_batch, end_batch) {
      print(sprintf("Batch %i/%i", which(init_batch == init_batches), length(init_batches)))
      get_integrated_gradients(input = x_input[init_batch:end_batch,,,], 
                               site = output_neuron, 
                               model = model,
                               baseline = baseline, 
                               num_steps = num_steps)      
    }) %>% abind::abind(along = 1)
  }
  
  ### Verify the axiom of completeness
  # axiom_completeness(input = x_input,
  #                    site = output_neuron,
  #                    model = model,
  #                    baseline = NULL,
  #                    integrated.gradients = array_int_grads)
  
  ### Store the integrated gradients in a climate4R object
  if (getShape(x, "var") > 1) {
    if (length(dim(x_input)) > 3) { ### 3D predictor objects ("var", "lat", "lon")
      if (model.info[["channels"]] == "last") {
        array_int_grads %<>% aperm(c(4,1,2,3))
      } else if (model.info[["channels"]] == "first") {
        array_int_grads %<>% aperm(c(2,1,3,4)) 
      }
    } else { ### 2D predictor objects ("var", "loc")
      if (model.info[["channels"]] == "last") {
        array_int_grads %<>% aperm(c(3,1,2))
      } else if (model.info[["channels"]] == "first") {
        array_int_grads %<>% aperm(c(2,1,3)) 
      }
    }
  }
  
  c4r_int_grads <- x
  c4r_int_grads$Data <- array_int_grads
  attr(c4r_int_grads$Data, "dimensions") <- attr(x$Data, "dimensions")
  
  ### Apply saliency aggregated function
  if (! is.null(saliency.fun)) c4r_int_grads %<>% climatology(clim.fun = saliency.fun)
  
  ### Return
  return(c4r_int_grads)  
}
