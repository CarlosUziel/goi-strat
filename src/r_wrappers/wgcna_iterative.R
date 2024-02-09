# Implements the WGCNA refinement algorithm proposed by XXXX et al.

# refine one WGCNA run by only using retained genes as long as 

refineWGCNA <- function(datExpr,
                        maxIter = 100,
                        weights = NULL,
                        checkMissingData = TRUE,
                        blocks = NULL,
                        maxBlockSize = 5000,
                        blockSizePenaltyPower = 5,
                        nPreclusteringCenters = as.integer(min(ncol(datExpr)/20, 100 * ncol(datExpr)/maxBlockSize)),
                        randomSeed = 54321,
                        loadTOM = FALSE,
                        corType = "pearson",
                        maxPOutliers = 1,
                        quickCor = 0,
                        pearsonFallback = "individual",
                        cosineCorrelation = FALSE,
                        power = 6,
                        networkType = "unsigned",
                        replaceMissingAdjacencies = FALSE,
                        TOMType = "signed",
                        TOMDenom = "min",
                        suppressTOMForZeroAdjacencies = FALSE,
                        suppressNegativeTOM = FALSE,
                        getTOMs = NULL,
                        saveTOMs = FALSE,
                        saveTOMFileBase = "blockwiseTOM",
                        deepSplit = 2,
                        detectCutHeight = 0.995,
                        minModuleSize = 20,
                        maxCoreScatter = NULL,
                        minGap = NULL,
                        maxAbsCoreScatter = NULL,
                        minAbsGap = NULL,
                        minSplitHeight = NULL,
                        minAbsSplitHeight = NULL,
                        useBranchEigennodeDissim = FALSE,
                        minBranchEigennodeDissim = mergeCutHeight,
                        stabilityLabels = NULL,
                        stabilityCriterion = c("Individual fraction", "Common fraction"),
                        minStabilityDissim = NULL,
                        pamStage = TRUE,
                        pamRespectsDendro = TRUE,
                        reassignThreshold = 0.05,
                        minCoreKME = 0.8,
                        minCoreKMESize = minModuleSize/3,
                        minKMEtoStay = 0.8,
                        mergeCutHeight = 0.15,
                        impute = TRUE,
                        trapErrors = TRUE,
                        nThreads = 0,
                        useInternalMatrixAlgebra = FALSE,
                        useCorOptionsThroughout = TRUE,
                        verbose = 0,
                        indent = 0, ...){
  # isolate the parameters and remove non blockwiseModule related
  require(WGCNA)
  params <-  c(as.list(environment()), list(...))

  # set numeric labels to true to avoid any problems

  params$numericLabels = TRUE

  # define some starting variables
  counter <- 0
  convergence <- FALSE
  print("Pruning: initial network estimate")
  
  # do the initial network estimate using the supplied parameters
  net <- do.call(blockwiseModules, params)
  
  # save all parameters in the net object for further reference
  net$params <- params
  
  # clone the network
  net1 <- net
  
  # main loop. Loop while iterations are not reached and genes can be pruned
  print("Entering refinement stage")
  while ((counter < maxIter) & (!convergence)){
    counter <- counter+1
    print(paste("Pruning iteration", counter))
    nGenes <- ncol(params$datExpr)
    
    # remove unassigned genes from the expression matrix and calculate the new expression matrix
    params$datExpr <- params$datExpr[,net1$colors > 0]
    nGenes1 <- ncol(params$datExpr)
    
    print(paste("pruned genes:", 
                nGenes-nGenes1,
                "remaining genes:", 
                nGenes1))
    
    convergence <- TRUE
    
    # are there any genes left ? If yes
    if (nGenes1 > 0){
      
      # test if there are good genes left
      goodGenes <- try(goodGenes(params$datExpr),
                       silent = TRUE)
      
      # if yes and Genes have been pruned - continue
      if((nGenes > nGenes1) & (class(goodGenes) != "try-error")){
        net1 <- do.call(blockwiseModules, params)
        convergence <- FALSE
      }
    }
    
    # test if there are enough genes left to detect modules
    goodGenes <- try(goodGenes(params$datExpr),
                     silent = TRUE)
    # if genes have been pruned, recalculate the network else set convergence to true
    
  }
  print(paste("Convergence reached after", counter, "iteration"))
  net$refinedColors <- rep(0, ncol(net$params$datExpr))
  names(net$refinedColors) <- colnames(net$params$datExpr)
  net$refinedColors[names(net1$colors)] <- net1$colors
  
  # return the extended network object
  return(net)
}

#### Iterative WGCNA - refine networks until the network stabilizes

iterativeWGCNA <- function(datExpr,
                           maxIter = 100,
                           weights = NULL,
                           checkMissingData = TRUE,
                           blocks = NULL,
                           maxBlockSize = 5000,
                           blockSizePenaltyPower = 5,
                           nPreclusteringCenters = as.integer(min(ncol(datExpr)/20, 100 * ncol(datExpr)/maxBlockSize)),
                           randomSeed = 54321,
                           loadTOM = FALSE,
                           corType = "pearson",
                           maxPOutliers = 1,
                           quickCor = 0,
                           pearsonFallback = "individual",
                           cosineCorrelation = FALSE,
                           power = 6,
                           networkType = "unsigned",
                           replaceMissingAdjacencies = FALSE,
                           TOMType = "signed",
                           TOMDenom = "min",
                           suppressTOMForZeroAdjacencies = FALSE,
                           suppressNegativeTOM = FALSE,
                           getTOMs = NULL,
                           saveTOMs = FALSE,
                           saveTOMFileBase = "blockwiseTOM",
                           deepSplit = 2,
                           detectCutHeight = 0.995,
                           minModuleSize = 20,
                           maxCoreScatter = NULL,
                           minGap = NULL,
                           maxAbsCoreScatter = NULL,
                           minAbsGap = NULL,
                           minSplitHeight = NULL,
                           minAbsSplitHeight = NULL,
                           useBranchEigennodeDissim = FALSE,
                           minBranchEigennodeDissim = mergeCutHeight,
                           stabilityLabels = NULL,
                           stabilityCriterion = c("Individual fraction", "Common fraction"),
                           minStabilityDissim = NULL,
                           pamStage = TRUE,
                           pamRespectsDendro = TRUE,
                           reassignThreshold = 0.05,
                           minCoreKME = 0.8,
                           minCoreKMESize = minModuleSize/3,
                           minKMEtoStay = 0.8,
                           mergeCutHeight = 0.15,
                           impute = TRUE,
                           trapErrors = TRUE,
                           numericLabels = FALSE,
                           nThreads = 0,
                           useInternalMatrixAlgebra = FALSE,
                           useCorOptionsThroughout = TRUE,
                           verbose = 0,
                           indent = 0, ...){

  # extract the variables related to WGCNA
  require(WGCNA)
  params <-  c(as.list(environment()), list(...))
  # define some starting variables
  counter <- 0
  convergence <- FALSE
  refinedColors <- c()
  max <- 0
  
  # initial network estimation
  print("Initial network estimation")
  net <- do.call(refineWGCNA, params)
  
  # keep the original network
  net1 <- net
  
  # iterate over the networks
  while (counter < maxIter & !convergence) {
    counter <- counter + 1
    print(paste("Iteration", counter))
    
    # calculate non assigned genes
    unassignedGenes <- net1$refinedColors == 0
    print(paste("number of unassigned genes:", sum(unassignedGenes)))
    
    if (sum(unassignedGenes) <= 3){
      break
    }

    # extract the genes assigned to a module
    colors <- net1$refinedColors[!unassignedGenes]
    
    # add the maximum of the last iteration
    colors <- colors + max
    
    # define the new maximum
    max <- max(colors)
    
    # add the new colors to the color vector
    refinedColors <- c(refinedColors, colors)
    
    # remove assigned genes
    params$datExpr <- params$datExpr[,unassignedGenes]
    
    # repeat network refinement

    net1 <- do.call(refineWGCNA, params)
    
    # calculate non assigned genes
    
    # set convergence to TRUE if no modules could be detected, i.e. there are no assigned genes in the new network
    if (sum(net1$refinedColors) == 0){
      convergence <- TRUE
    }
  }
  
  # assemble results into one colorvector
  
  # set all genes to grey
  finalColors <- rep(0, ncol(datExpr))
  names(finalColors) <- colnames(datExpr)
  
  # assign the result of the iteration
  finalColors[names(refinedColors)] <- refinedColors
  
  # reassign genes to closer modules
  KMEs <- WGCNA::moduleEigengenes(datExpr, finalColors)
  cor <- corAndPvalue(datExpr, KMEs$eigengenes)$cor
  cor[cor<minKMEtoStay] <- 0
  cor[rowSums(cor) == 0, "ME0"] <- 1
  finalColors  <- apply(cor, 
                       1,
                       which.max) - 1
  # refine the modules by assigning the gene to the module with the highest module membership
  # calculate the module membership
  
  finalColors <- mergeCloseModules(exprData = datExpr,
                                   colors = finalColors,
                                   cutHeight = mergeCutHeight,
                                   trapErrors = trapErrors,
                                   relabel = TRUE)
  net$refinedColors <- finalColors$colors[colnames(net$params$datExpr)]
  # labels as numeric or character ?
  if (!numericLabels){
    net$colors <- labels2colors(net$colors)
    net$refinedColors <- labels2colors(net$refinedColors)
  }
  
  # return the result
  return(net)
}
