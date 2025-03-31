#Funkcje

compute_auc <- function(y_true, y_prob, positive_class) {
  y_binary <- ifelse(y_true == positive_class, 1, 0)
  df <- data.frame(y = y_binary, prob = as.numeric(as.character(y_prob)))
  df <- df[order(-df$prob), ]
  
  nPos <- sum(df$y)
  nNeg <- nrow(df) - nPos
  
  TPR <- c(0)
  FPR <- c(0)
  
  for(i in 1:nrow(df)) {
    thresh <- df$prob[i]
    preds <- ifelse(y_prob >= thresh, 1, 0)
    TP <- sum(preds == 1 & y_binary == 1)
    FP <- sum(preds == 1 & y_binary == 0)
    TPR <- c(TPR, ifelse(nPos == 0, 0, TP / nPos))
    FPR <- c(FPR, ifelse(nNeg == 0, 0, FP / nNeg))
  }
  TPR <- c(TPR, 1)
  FPR <- c(FPR, 1)
  
  ord <- order(FPR)
  FPR <- FPR[ord]
  TPR <- TPR[ord]
  
  auc <- sum(diff(FPR) * (TPR[-1] + TPR[-length(TPR)]) / 2)
  return(auc)
}

compute_multiclass_auc <- function(y_true, probs, classes = NULL) {
  if(is.null(colnames(probs)) && !is.null(classes)) {
    colnames(probs) <- classes
  }
  if(is.null(classes)) classes <- colnames(probs)
  aucs <- numeric(length(classes))
  for(i in seq_along(classes)){
    pos_class <- classes[i]
    aucs[i] <- compute_auc(y_true, probs[, pos_class], pos_class)
  }
  return(mean(aucs, na.rm = TRUE))
}



modelOcena <- function(y_true, y_pred, positive_class = NULL) {
  if (is.numeric(y_true)) {
    RMSE <- sqrt(mean((y_true - y_pred)^2))
    MAE  <- mean(abs(y_true - y_pred))
    MSE  <- mean((y_true - y_pred)^2)
    MAPE <- mean(ifelse(y_true == 0, NA, abs((y_true - y_pred) / y_true)), na.rm = TRUE) * 100
    return(list(RMSE = RMSE, MAE = MAE, MSE = MSE, MAPE = MAPE))
  }
  if (is.factor(y_true)) {
    if (length(levels(y_true)) == 2) {
      if (is.null(positive_class)) {
        positive_class <- levels(y_true)[2]
      }
      if (is.list(y_pred) && !is.null(y_pred$probs)) {
        if (is.matrix(y_pred$probs) || is.data.frame(y_pred$probs)) {
          prob_positive <- y_pred$probs[, positive_class]
        } else {
          prob_positive <- y_pred$probs[positive_class]
        }
        pred_class <- y_pred$class
      } else if (is.data.frame(y_pred) || is.matrix(y_pred)) {
        y_pred_mat <- as.data.frame(y_pred)
        if (all(levels(y_true) %in% colnames(y_pred_mat))) {
          prob_positive <- y_pred_mat[[positive_class]]
          pred_class <- factor(apply(y_pred_mat, 1, function(x) names(x)[which.max(x)]),
                               levels = levels(y_true))
        } else if ("Klasa" %in% colnames(y_pred_mat) && positive_class %in% colnames(y_pred_mat)) {
          prob_positive <- y_pred_mat[[positive_class]]
          pred_class <- y_pred_mat$Klasa
        } else {
          stop("Brak kolumny z prawdopodobienstwem dla klasy pozytywnej.")
        }
      } else if (is.numeric(y_pred)) {
        prob_positive <- y_pred
        pred_class <- factor(ifelse(y_pred >= 0.5, positive_class, levels(y_true)[1]),
                             levels = levels(y_true))
      } else if (is.factor(y_pred)) {
        pred_class <- y_pred
        prob_positive <- NA
      } else {
        stop("Nieobslugiwany format predykcji.")
      }
      
      TP <- sum(y_true == positive_class & pred_class == positive_class)
      TN <- sum(y_true != positive_class & pred_class != positive_class)
      FP <- sum(y_true != positive_class & pred_class == positive_class)
      FN <- sum(y_true == positive_class & pred_class != positive_class)
      
      accuracy    <- (TP + TN) / (TP + TN + FP + FN)
      precision   <- ifelse((TP + FP) == 0, NA, TP / (TP + FP))
      sensitivity <- ifelse((TP + FN) == 0, NA, TP / (TP + FN))
      specificity <- ifelse((TN + FP) == 0, NA, TN / (TN + FP))
      F1          <- ifelse(is.na(precision) | is.na(sensitivity) | (precision + sensitivity) == 0,
                            NA, 2 * precision * sensitivity / (precision + sensitivity))
      youden      <- sensitivity + specificity - 1
      
      auc <- if (!all(is.na(prob_positive))) compute_auc(y_true, prob_positive, positive_class) else NA
      
      return(list(Accuracy = accuracy,
                  Precision = precision,
                  Sensitivity = sensitivity,
                  Specificity = specificity,
                  F1 = F1,
                  YoudenIndex = youden,
                  AUC = auc,
                  ConfusionMatrix = table(Predicted = pred_class, Actual = y_true)))
    } else {
      classes <- levels(y_true)
      if (is.list(y_pred) && !is.null(y_pred$probs)) {
        probs_mat  <- y_pred$probs
        if (is.null(colnames(probs_mat))) {
          colnames(probs_mat) <- classes
        }
        pred_class <- y_pred$class
      } else if (is.data.frame(y_pred) || is.matrix(y_pred)) {
        probs_mat <- as.matrix(y_pred)
        if (!all(classes %in% colnames(probs_mat))) {
          stop("Brak prawdopodobienstw dla wszystkich klas.")
        }
        pred_class <- factor(apply(probs_mat, 1, function(x) names(x)[which.max(x)]), levels = classes)
      } else if (is.numeric(y_pred)) {
        stop("Dla klasyfikacji wieloklasowej oczekiwane sa prawdopodobienstwa.")
      } else if (is.factor(y_pred)) {
        pred_class <- y_pred
        probs_mat <- NA
      } else {
        stop("Nieobslugiwany format predykcji.")
      }
      
      conf_matrix <- matrix(0, nrow = length(classes), ncol = length(classes),
                            dimnames = list(Predicted = classes, Actual = classes))
      for (i in seq_along(y_true)) {
        true_class <- as.character(y_true[i])
        pred_class_i <- as.character(pred_class[i])
        if (!(pred_class_i %in% classes) || !(true_class %in% classes)) {
          warning("Wartosc predykcji lub etykiety nie nalezy do oczekiwanych poziomów!")
          next
        }
        conf_matrix[pred_class_i, true_class] <- conf_matrix[pred_class_i, true_class] + 1
      }
      overall_accuracy <- sum(diag(conf_matrix)) / sum(conf_matrix)
      precision_per_class <- numeric(length(classes))
      sensitivity_per_class <- numeric(length(classes))
      specificity_per_class <- numeric(length(classes))
      f1_per_class <- numeric(length(classes))
      names(precision_per_class) <- classes
      names(sensitivity_per_class) <- classes
      names(specificity_per_class) <- classes
      names(f1_per_class) <- classes
      for (cl in classes) {
        TP <- conf_matrix[cl, cl]
        FP <- sum(conf_matrix[cl, ]) - TP
        FN <- sum(conf_matrix[, cl]) - TP
        TN <- sum(conf_matrix) - (TP + FP + FN)
        precision_per_class[cl] <- ifelse((TP + FP) == 0, NA, TP / (TP + FP))
        sensitivity_per_class[cl] <- ifelse((TP + FN) == 0, NA, TP / (TP + FN))
        specificity_per_class[cl] <- ifelse((TN + FP) == 0, NA, TN / (TN + FP))
        f1_per_class[cl] <- ifelse(is.na(precision_per_class[cl]) || is.na(sensitivity_per_class[cl]) ||
                                     (precision_per_class[cl] + sensitivity_per_class[cl]) == 0,
                                   NA,
                                   2 * precision_per_class[cl] * sensitivity_per_class[cl] /
                                     (precision_per_class[cl] + sensitivity_per_class[cl]))
      }
      macro_precision   <- mean(precision_per_class, na.rm = TRUE)
      macro_sensitivity <- mean(sensitivity_per_class, na.rm = TRUE)
      macro_specificity <- mean(specificity_per_class, na.rm = TRUE)
      macro_f1          <- mean(f1_per_class, na.rm = TRUE)
      
      multi_auc <- if (!is.na(probs_mat)[1]) compute_multiclass_auc(y_true, probs_mat, classes) else NA
      
      return(list(ConfusionMatrix = conf_matrix,
                  OverallAccuracy = overall_accuracy,
                  Precision_per_class = precision_per_class,
                  Sensitivity_per_class = sensitivity_per_class,
                  Specificity_per_class = specificity_per_class,
                  F1_per_class = f1_per_class,
                  MacroPrecision = macro_precision,
                  MacroSensitivity = macro_sensitivity,
                  MacroSpecificity = macro_specificity,
                  MacroF1 = macro_f1,
                  Multi_AUC = multi_auc))
    }
  }
  stop("Nieobslugiwany typ danych wejsciowych w modelOcena.")
}



distance_euclidean <- function(x, y) {
  sqrt(sum((x - y)^2))
}

distance_hamming <- function(x, y) {
  sum(x != y)
}

distance_gower <- function(x, y, ranges, types) {
  d <- numeric(length(x))
  for (i in seq_along(x)) {
    if (types[i] == "numeric") {
      if (ranges[i] == 0) {
        d[i] <- 0
      } else {
        d[i] <- abs(x[i] - y[i]) / ranges[i]
      }
    } else {
      d[i] <- ifelse(x[i] == y[i], 0, 1)
    }
  }
  return(mean(d))
}

KNNtrain <- function(X, y, k = 5) {
  if (!is.data.frame(X) && !is.matrix(X))
    stop("X musi byc data.frame lub macierza.")
  X <- as.data.frame(X)
  
  norm_info <- list()
  for (col in names(X)) {
    if (is.numeric(X[[col]])) {
      if (any(is.na(X[[col]])))
        X[[col]][is.na(X[[col]])] <- median(X[[col]], na.rm = TRUE)
      orig_min <- min(X[[col]])
      orig_max <- max(X[[col]])
      range_val <- orig_max - orig_min
      if (range_val == 0) {
        X[[col]] <- 0.5
        norm_info[[col]] <- list(min = orig_min, max = orig_max)
      } else {
        X[[col]] <- (X[[col]] - orig_min) / range_val
        norm_info[[col]] <- list(min = orig_min, max = orig_max, range = range_val)
      }
    } else if (is.factor(X[[col]])) {
      if (any(is.na(X[[col]])))
        X[[col]][is.na(X[[col]])] <- get_mode(X[[col]])
      X[[col]] <- as.factor(X[[col]])
    }
  }
  
  types <- sapply(X, function(col) if(is.numeric(col)) "numeric" else "factor")
  if (all(types == "numeric")) {
    metric <- "euclidean"
  } else if (all(types == "factor")) {
    metric <- "hamming"
  } else {
    metric <- "gower"
    for (col in names(X)[types=="numeric"]) {
      norm_info[[col]]$range <- 1
    }
    for (col in names(X)[types=="factor"]) {
      norm_info[[col]]$range <- 1
    }
  }
  
  return(list(X = X, y = y, k = k, metric = metric, norm_info = norm_info, col_types = types))
}

KNNpredict <- function(model, Xnew) {
  if (!is.data.frame(Xnew) && !is.matrix(Xnew))
    stop("Xnew musi byc data.frame lub macierza.")
  Xnew <- as.data.frame(Xnew)
  required_cols <- names(model$X)
  if (!all(required_cols %in% names(Xnew)))
    stop("Xnew nie zawiera wszystkich wymaganych kolumn.")
  
  for (col in required_cols) {
    if (model$col_types[col] == "numeric") {
      if (any(is.na(Xnew[[col]])))
        Xnew[[col]][is.na(Xnew[[col]])] <- median(Xnew[[col]], na.rm = TRUE)
      info <- model$norm_info[[col]]
      range_val <- info$max - info$min
      if (range_val == 0) {
        Xnew[[col]] <- 0.5
      } else {
        Xnew[[col]] <- (Xnew[[col]] - info$min) / range_val
      }
    } else if (model$col_types[col] == "factor") {
      if (any(is.na(Xnew[[col]])))
        Xnew[[col]][is.na(Xnew[[col]])] <- get_mode(Xnew[[col]])
      Xnew[[col]] <- factor(Xnew[[col]], levels = levels(model$X[[col]]))
    }
  }
  
  nTrain <- nrow(model$X)
  nPred <- nrow(Xnew)
  predictions <- vector("list", nPred)
  for (i in 1:nPred) {
    dists <- numeric(nTrain)
    for (j in 1:nTrain) {
      x_train <- model$X[j, ]
      x_new   <- Xnew[i, ]
      if (model$metric == "euclidean") {
        dists[j] <- distance_euclidean(as.numeric(x_train), as.numeric(x_new))
      } else if (model$metric == "hamming") {
        dists[j] <- distance_hamming(as.character(x_train), as.character(x_new))
      } else if (model$metric == "gower") {
        x_train_vec <- unlist(x_train, use.names = FALSE)
        x_new_vec   <- unlist(x_new, use.names = FALSE)
        col_names   <- names(model$X)
        col_ranges  <- sapply(col_names, function(col) model$norm_info[[col]]$range)
        types_vec   <- model$col_types
        dists[j] <- distance_gower(x_train_vec, x_new_vec, col_ranges, types_vec)
      }
    }
    k_indices <- order(dists)[1:model$k]
    neighbor_y <- model$y[k_indices]
    if (is.numeric(model$y)) {
      predictions[[i]] <- mean(neighbor_y)
    } else if (is.factor(model$y)) {
      classes <- levels(model$y)
      counts <- table(factor(neighbor_y, levels = classes))
      probs <- counts / sum(counts)
      probs_full <- sapply(classes, function(cl) {
        if(cl %in% names(probs)) probs[cl] else 0
      })
      names(probs_full) <- classes
      pred_class <- names(which.max(probs_full))
      predictions[[i]] <- list(probs = probs_full, class = factor(pred_class, levels = classes))
    }
  }
  if (is.numeric(model$y)) {
    return(unlist(predictions))
  } else if (is.factor(model$y)) {
    probs_mat <- do.call(rbind, lapply(predictions, function(x) x$probs))
    class_vec <- factor(sapply(predictions, function(x) x$class), levels = levels(model$y))
    return(list(probs = probs_mat, class = class_vec))
  }
}


cross_validate_knn <- function(X, y, k_values, folds = 10) {
  set.seed(123)
  best_loss <- Inf
  best_k <- NULL
  folds_idx <- caret::createFolds(y, k = folds, list = TRUE)
  
  if (is.numeric(y)) {
    loss_function <- function(res) { res$RMSE }
  } else if (is.factor(y)) {
    if (length(levels(y)) == 2) {
      loss_function <- function(res) { 1 - res$Accuracy }
    } else {
      loss_function <- function(res) { 1 - res$OverallAccuracy }
    }
  } else {
    stop("Unsupported target type")
  }
  
  for (k in k_values) {
    losses <- numeric(length(folds_idx))
    for (i in seq_along(folds_idx)) {
      fold <- folds_idx[[i]]
      train_idx <- setdiff(seq_len(nrow(X)), fold)
      test_idx  <- fold
      train_X <- X[train_idx, , drop = FALSE]
      test_X  <- X[test_idx, , drop = FALSE]
      train_y <- y[train_idx]
      test_y  <- y[test_idx]
      model <- KNNtrain(train_X, train_y, k)
      preds <- KNNpredict(model, test_X)
      losses[i] <- loss_function(modelOcena(test_y, preds))
    }
    mean_loss <- mean(losses, na.rm = TRUE)
    if (mean_loss < best_loss) {
      best_loss <- mean_loss
      best_k <- k
    }
  }
  return(list(best_k = best_k, best_loss = best_loss))
}



sigmoid <- function(x) { 1/(1+exp(-x)) }
d_sigmoid <- function(x) { x*(1-x) }
softmax <- function(x) {
  exp_x <- exp(x - max(x))
  return(exp_x / sum(exp_x))
}
identity_activation <- function(x) { x }
d_identity <- function(x) { rep(1, length(x)) }

forwardPropagation2 <- function(X, weights, type) {
  L <- length(weights)
  a <- X
  activations <- list(a)
  zs <- list()
  for (l in 1:L) {
    a <- cbind(1, a)
    z <- a %*% weights[[l]]
    zs[[l]] <- z
    if (l < L) {
      a <- sigmoid(z)
    } else {
      if (type == "regression") {
        a <- identity_activation(z)
      } else if (type == "binary") {
        a <- sigmoid(z)
      } else if (type == "multiclass") {
        a <- t(apply(z, 1, softmax))
      }
    }
    activations[[l+1]] <- a
  }
  return(list(activations = activations, zs = zs))
}

backwardPropagation2 <- function(y, weights, forward, type) {
  activations <- forward$activations
  zs <- forward$zs
  L <- length(weights)
  m <- nrow(activations[[1]])
  deltas <- vector("list", L)
  grads <- vector("list", L)
  
  a_out <- activations[[L+1]]
  delta_L <- (a_out - y)
  deltas[[L]] <- delta_L
  
  if (L >= 2) {
    for (l in (L-1):1) {
      W_next <- weights[[l+1]]
      W_next_no_bias <- W_next[-1, , drop = FALSE]
      a_hidden <- sigmoid(zs[[l]])
      delta_l <- (deltas[[l+1]] %*% t(W_next_no_bias)) * d_sigmoid(a_hidden)
      deltas[[l]] <- delta_l
    }
  }
  
  for (l in 1:L) {
    a_l <- cbind(1, activations[[l]])
    grads[[l]] <- t(a_l) %*% deltas[[l]] / m
  }
  return(grads)
}

trainNN2 <- function(X, y, type = c("regression", "binary", "multiclass"),
                     h = c(5), lr = 0.01, iter = 5000, seed = 123) {
  type <- match.arg(type)
  set.seed(seed)
  X <- as.matrix(X)
  for (j in 1:ncol(X)) {
    if (any(is.na(X[,j])))
      X[,j][is.na(X[,j])] <- median(X[,j], na.rm = TRUE)
    rng <- max(X[,j]) - min(X[,j])
    if (rng == 0) {
      X[,j] <- 0.5
    } else {
      X[,j] <- (X[,j] - min(X[,j])) / rng
    }
  }
  m <- nrow(X)
  d <- ncol(X)
  
  if (type == "regression") {
    out_dim <- 1
    y <- as.matrix(y)
  } else if (type == "binary") {
    out_dim <- 1
    if (is.factor(y)) {
      y <- as.matrix(as.numeric(y) - 1)
    } else {
      y <- as.matrix(y)
    }
  } else if (type == "multiclass") {
    class_levels <- levels(y)
    out_dim <- length(class_levels)
    y <- model.matrix(~ y - 1)
  }
  
  weights <- list()
  weights[[1]] <- matrix(runif((d+1)*h[1], min = -1, max = 1), nrow = d+1, ncol = h[1])
  if (length(h) > 1) {
    for (i in 2:length(h)) {
      weights[[i]] <- matrix(runif((h[i-1]+1)*h[i], min = -1, max = 1), nrow = h[i-1]+1, ncol = h[i])
    }
  }
  last_input <- if (length(h) == 0) d else h[length(h)]
  weights[[length(weights)+1]] <- matrix(runif((last_input+1)*out_dim, min = -1, max = 1),
                                         nrow = last_input+1, ncol = out_dim)
  
  for (i in 1:iter) {
    forward <- forwardPropagation2(X, weights, type)
    grads <- backwardPropagation2(y, weights, forward, type)
    for (l in 1:length(weights)) {
      weights[[l]] <- weights[[l]] - lr * grads[[l]]
    }
    #if (i %% 1000 == 0) cat("Iteracja:", i, "\n")
  }
  forward_final <- forwardPropagation2(X, weights, type)
  y_hat <- forward_final$activations[[length(weights)+1]]
  if(type == "multiclass"){
    return(list(weights = weights, y_hat = y_hat, class_levels = class_levels))
  } else {
    return(list(weights = weights, y_hat = y_hat))
  }
}

predictNN2 <- function(X, NN, type = c("regression", "binary", "multiclass")) {
  type <- match.arg(type)
  X <- as.matrix(X)
  forward <- forwardPropagation2(X, NN$weights, type)
  y_hat <- forward$activations[[length(NN$weights)+1]]
  if (type == "binary") {
    probs <- as.vector(y_hat)
    pred_class <- ifelse(probs >= 0.5, 1, 0)
    return(list(probs = probs, class = factor(pred_class, levels = c(0,1))))
  } else if (type == "multiclass") {
    if(!is.null(NN$class_levels)){
      colnames(y_hat) <- NN$class_levels
    }
    probs <- y_hat
    pred_indices <- apply(probs, 1, which.max)
    pred_class <- factor(NN$class_levels[pred_indices], levels = NN$class_levels)
    return(list(probs = probs, class = pred_class))
  } else {
    return(unlist(y_hat))
  }
}



get_mode <- function(v) {
  uniqv <- unique(v)
  uniqv[which.max(tabulate(match(v, uniqv)))]
}

impurity_value <- function(y, type) {
  n <- length(y)
  if(n == 0) return(0)
  if (type == "regression") {
    m <- mean(y)
    return(sum((y - m)^2))
  } else if (type == "classification") {
    p <- table(y) / n
    return(1 - sum(p^2))
  }
}

find_best_split_for_var <- function(x, y, type, minobs) {
  if (is.factor(x)) x <- as.numeric(x)
  x_unique <- sort(unique(x))
  if (length(x_unique) < 2) return(NULL)
  best_gain <- -Inf
  best_point <- NA
  parent_imp <- impurity_value(y, type)
  best_left_idx <- best_right_idx <- NULL
  for (split_point in x_unique[-length(x_unique)]) {
    left_idx <- which(x <= split_point)
    right_idx <- which(x > split_point)
    if (length(left_idx) < minobs || length(right_idx) < minobs) next
    left_imp <- impurity_value(y[left_idx], type)
    right_imp <- impurity_value(y[right_idx], type)
    n <- length(y)
    gain <- parent_imp - (length(left_idx)/n * left_imp + length(right_idx)/n * right_imp)
    if (gain > best_gain) {
      best_gain <- gain
      best_point <- split_point
      best_left_idx <- left_idx
      best_right_idx <- right_idx
    }
  }
  if (is.infinite(best_gain) || best_gain <= 0) return(NULL)
  return(list(gain = best_gain, point = best_point,
              left_idx = best_left_idx, right_idx = best_right_idx))
}

find_best_split <- function(X, y, type, minobs) {
  best_overall_gain <- -Inf
  best_split <- NULL
  best_var <- NULL
  for (var in names(X)) {
    res <- find_best_split_for_var(X[[var]], y, type, minobs)
    if (!is.null(res) && res$gain > best_overall_gain) {
      best_overall_gain <- res$gain
      best_split <- res
      best_var <- var
    }
  }
  if (is.null(best_split)) return(NULL)
  best_split$var <- best_var
  return(best_split)
}

myDecisionTree <- function(X, y, type = c("regression", "classification"), 
                           max_depth = 5, minobs = 5) {
  type <- match.arg(type)
  for (col in names(X)) {
    if (is.numeric(X[[col]])) {
      if (any(is.na(X[[col]])))
        X[[col]][is.na(X[[col]])] <- median(X[[col]], na.rm = TRUE)
    } else if (is.factor(X[[col]])) {
      if (any(is.na(X[[col]])))
        X[[col]][is.na(X[[col]])] <- get_mode(X[[col]])
      X[[col]] <- as.factor(X[[col]])
    }
  }
  if (max_depth == 0 || nrow(X) < minobs || impurity_value(y, type) == 0) {
    if (type == "regression") {
      pred <- mean(y)
      return(list(is_leaf = TRUE, prediction = pred, n = nrow(X)))
    } else {
      tbl <- table(y)
      probs <- as.numeric(tbl) / sum(tbl)
      names(probs) <- names(tbl)
      pred <- names(which.max(tbl))
      return(list(is_leaf = TRUE, prediction = pred, probs = probs, n = nrow(X)))
    }
  }
  split <- find_best_split(X, y, type, minobs)
  if (is.null(split)) {
    if (type == "regression") {
      pred <- mean(y)
      return(list(is_leaf = TRUE, prediction = pred, n = nrow(X)))
    } else {
      tbl <- table(y)
      probs <- as.numeric(tbl) / sum(tbl)
      names(probs) <- names(tbl)
      pred <- names(which.max(tbl))
      return(list(is_leaf = TRUE, prediction = pred, probs = probs, n = nrow(X)))
    }
  }
  left_tree <- myDecisionTree(X[split$left_idx, , drop = FALSE],
                              y[split$left_idx], type, max_depth - 1, minobs)
  right_tree <- myDecisionTree(X[split$right_idx, , drop = FALSE],
                               y[split$right_idx], type, max_depth - 1, minobs)
  if (type == "regression") {
    pred <- mean(y)
    return(list(is_leaf = FALSE,
                split_var = split$var,
                split_point = split$point,
                gain = split$gain,
                prediction = pred,
                left = left_tree,
                right = right_tree,
                n = nrow(X)))
  } else {
    tbl <- table(y)
    probs <- as.numeric(tbl) / sum(tbl)
    names(probs) <- names(tbl)
    pred <- names(which.max(tbl))
    return(list(is_leaf = FALSE,
                split_var = split$var,
                split_point = split$point,
                gain = split$gain,
                prediction = pred,
                probs = probs,
                left = left_tree,
                right = right_tree,
                n = nrow(X)))
  }
}

predict_tree <- function(tree, Xnew) {
  n <- nrow(Xnew)
  preds <- vector("list", n)
  for (i in 1:n) {
    node <- tree
    while (!node$is_leaf) {
      value <- Xnew[i, node$split_var]
      if (is.factor(value)) value <- as.numeric(value)
      if (value <= node$split_point) {
        node <- node$left
      } else {
        node <- node$right
      }
    }
    if (is.numeric(node$prediction)) {
      preds[[i]] <- node$prediction
    } else {
      preds[[i]] <- list(probs = node$probs, class = factor(node$prediction, levels = names(node$probs)))
    }
  }
  if (is.numeric(tree$prediction)) {
    return(unlist(preds))
  } else {
    probs_mat <- do.call(rbind, lapply(preds, function(x) x$probs))
    class_vec <- factor(sapply(preds, function(x) x$class), levels = names(preds[[1]]$probs))
    return(list(probs = probs_mat, class = class_vec))
  }
}


cross_validate_tree <- function(X, y, depths = 1:5, minobs, type, folds = 5) {
  set.seed(123)
  best_loss <- Inf
  best_depth <- NULL
  cv_times <- c()
  if (is.numeric(y)) {
    loss_function <- function(res) { res$RMSE }
  } else if (is.factor(y)) {
    if (length(levels(y)) == 2) {
      loss_function <- function(res) { 1 - res$Accuracy }
    } else {
      loss_function <- function(res) { 1 - res$OverallAccuracy }
    }
  }
  
  for (d in depths) {
    losses <- c()
    start_fold <- Sys.time()
    folds_idx <- caret::createFolds(y, k = folds, list = TRUE)
    for (i in seq_along(folds_idx)) {
      fold <- folds_idx[[i]]
      train_idx <- setdiff(seq_len(nrow(X)), fold)
      test_idx <- fold
      tree_model <- myDecisionTree(X[train_idx, , drop = FALSE], y[train_idx],
                                   type = type, max_depth = d, minobs = minobs)
      preds <- predict_tree(tree_model, X[test_idx, , drop = FALSE])
      res <- modelOcena(y[test_idx], preds)
      losses[i] <- loss_function(res)
    }
    elapsed <- Sys.time() - start_fold
    cv_times <- c(cv_times, elapsed)
    mean_loss <- mean(losses, na.rm = TRUE)
    if (mean_loss < best_loss) {
      best_loss <- mean_loss
      best_depth <- d
    }
  }
  return(list(best_depth = best_depth, best_loss = best_loss, cv_time = mean(cv_times)))
}



plotConfMatrix <- function(conf_matrix, title = "Confusion Matrix") {
  library(reshape2)
  library(ggplot2)
  cm_df <- melt(conf_matrix)
  colnames(cm_df) <- c("Predicted", "Actual", "Count")
  ggplot(cm_df, aes(x = Actual, y = Predicted, fill = Count)) +
    geom_tile(color = "white") +
    geom_text(aes(label = Count)) +
    scale_fill_gradient(low = "white", high = "blue") +
    ggtitle(title) +
    theme_minimal()
}

plotTree <- function(tree) {
  draw_node <- function(node, depth = 0) {
    prefix <- paste(rep("  ", depth), collapse = "")
    if (node$is_leaf) {
      cat(prefix, "Leaf: prediction =", node$prediction, "\n", sep = "")
    } else {
      cat(prefix, "Node: split_var =", node$split_var, ", split_point =", round(node$split_point, 2), "\n", sep = "")
      draw_node(node$left, depth + 1)
      draw_node(node$right, depth + 1)
    }
  }
  draw_node(tree)
}


cv_tune_builtin_knn <- function(formula, data, k_values, folds = 5) {
  library(caret)
  results <- list()
  mf <- model.frame(formula, data)
  y <- mf[[1]]
  folds_idx <- caret::createFolds(y, k = folds, list = TRUE)
  
  for (k in k_values) {
    if (is.numeric(y)) {
      met_train <- list(MAE = c(), MSE = c(), MAPE = c())
      met_val   <- list(MAE = c(), MSE = c(), MAPE = c())
    } else {
      met_train <- list(Accuracy = c(), Sensitivity = c(), Specificity = c(), F1 = c())
      met_val   <- list(Accuracy = c(), Sensitivity = c(), Specificity = c(), F1 = c())
    }
    for (fold in folds_idx) {
      train_data <- data[-fold, ]
      val_data <- data[fold, ]
      if (is.numeric(y)) {
        tr_ctrl <- trainControl(method = "none")
        model <- train(formula, data = train_data, method = "knn",
                       tuneGrid = data.frame(k = k), trControl = tr_ctrl, preProcess = "scale")
        pred_train <- predict(model, newdata = train_data, type = "raw")
        pred_val   <- predict(model, newdata = val_data, type = "raw")
      } else {
        tr_ctrl <- trainControl(method = "none", classProbs = TRUE)
        model <- train(formula, data = train_data, method = "knn",
                       tuneGrid = data.frame(k = k), trControl = tr_ctrl, preProcess = "scale")
        pred_train <- predict(model, newdata = train_data, type = "prob")
        pred_val   <- predict(model, newdata = val_data, type = "prob")
      }
      res_train <- modelOcena(train_data[[as.character(formula[[2]])]], pred_train)
      res_val   <- modelOcena(val_data[[as.character(formula[[2]])]], pred_val)
      if (is.numeric(y)) {
        met_train$MAE <- c(met_train$MAE, res_train$MAE)
        met_train$MSE <- c(met_train$MSE, res_train$MSE)
        met_train$MAPE <- c(met_train$MAPE, res_train$MAPE)
        met_val$MAE   <- c(met_val$MAE, res_val$MAE)
        met_val$MSE   <- c(met_val$MSE, res_val$MSE)
        met_val$MAPE  <- c(met_val$MAPE, res_val$MAPE)
      } else {
        met_train$Accuracy <- c(met_train$Accuracy, res_train$Accuracy)
        met_train$Sensitivity <- c(met_train$Sensitivity, res_train$Sensitivity)
        met_train$Specificity <- c(met_train$Specificity, res_train$Specificity)
        met_train$F1 <- c(met_train$F1, res_train$F1)
        met_val$Accuracy <- c(met_val$Accuracy, res_val$Accuracy)
        met_val$Sensitivity <- c(met_val$Sensitivity, res_val$Sensitivity)
        met_val$Specificity <- c(met_val$Specificity, res_val$Specificity)
        met_val$F1 <- c(met_val$F1, res_val$F1)
      }
    }
    if (is.numeric(y)) {
      row <- data.frame(k = k,
                        MAE_train = mean(met_train$MAE, na.rm = TRUE),
                        MSE_train = mean(met_train$MSE, na.rm = TRUE),
                        MAPE_train = mean(met_train$MAPE, na.rm = TRUE),
                        MAE_val = mean(met_val$MAE, na.rm = TRUE),
                        MSE_val = mean(met_val$MSE, na.rm = TRUE),
                        MAPE_val = mean(met_val$MAPE, na.rm = TRUE))
      row$MAE_diff <- row$MAE_val - row$MAE_train
    } else {
      row <- data.frame(k = k,
                        Accuracy_train = mean(met_train$Accuracy, na.rm = TRUE),
                        Sensitivity_train = mean(met_train$Sensitivity, na.rm = TRUE),
                        Specificity_train = mean(met_train$Specificity, na.rm = TRUE),
                        F1_train = mean(met_train$F1, na.rm = TRUE),
                        Accuracy_val = mean(met_val$Accuracy, na.rm = TRUE),
                        Sensitivity_val = mean(met_val$Sensitivity, na.rm = TRUE),
                        Specificity_val = mean(met_val$Specificity, na.rm = TRUE),
                        F1_val = mean(met_val$F1, na.rm = TRUE))
      row$Accuracy_diff <- row$Accuracy_train - row$Accuracy_val
    }
    results[[length(results)+1]] <- row
  }
  results_df <- do.call(rbind, results)
  return(results_df)
}

cv_tune_builtin_tree <- function(formula, data, depths, minobs_values, folds = 5) {
  library(caret)
  results <- list()
  mf <- model.frame(formula, data)
  y <- mf[[1]]
  folds_idx <- caret::createFolds(y, k = folds, list = TRUE)
  
  for (minobs in minobs_values) {
    for (depth in depths) {
      if (is.numeric(y)) {
        met_train <- list(MAE = c(), MSE = c(), MAPE = c())
        met_val   <- list(MAE = c(), MSE = c(), MAPE = c())
      } else {
        met_train <- list(Accuracy = c(), Sensitivity = c(), Specificity = c(), F1 = c())
        met_val   <- list(Accuracy = c(), Sensitivity = c(), Specificity = c(), F1 = c())
      }
      for (fold in folds_idx) {
        train_data <- data[-fold, ]
        val_data <- data[fold, ]
        model <- rpart(formula, data = train_data,
                       control = rpart.control(minsplit = minobs, maxdepth = depth, cp = 0))
        if (is.numeric(y)) {
          pred_train <- predict(model, newdata = train_data)
          pred_val   <- predict(model, newdata = val_data)
        } else {
          pred_train <- predict(model, newdata = train_data, type = "prob")
          pred_val   <- predict(model, newdata = val_data, type = "prob")
        }
        res_train <- modelOcena(train_data[[as.character(formula[[2]])]], pred_train)
        res_val   <- modelOcena(val_data[[as.character(formula[[2]])]], pred_val)
        if (is.numeric(y)) {
          met_train$MAE <- c(met_train$MAE, res_train$MAE)
          met_train$MSE <- c(met_train$MSE, res_train$MSE)
          met_train$MAPE <- c(met_train$MAPE, res_train$MAPE)
          met_val$MAE   <- c(met_val$MAE, res_val$MAE)
          met_val$MSE   <- c(met_val$MSE, res_val$MSE)
          met_val$MAPE  <- c(met_val$MAPE, res_val$MAPE)
        } else {
          met_train$Accuracy <- c(met_train$Accuracy, res_train$Accuracy)
          met_train$Sensitivity <- c(met_train$Sensitivity, res_train$Sensitivity)
          met_train$Specificity <- c(met_train$Specificity, res_train$Specificity)
          met_train$F1 <- c(met_train$F1, res_train$F1)
          met_val$Accuracy <- c(met_val$Accuracy, res_val$Accuracy)
          met_val$Sensitivity <- c(met_val$Sensitivity, res_val$Sensitivity)
          met_val$Specificity <- c(met_val$Specificity, res_val$Specificity)
          met_val$F1 <- c(met_val$F1, res_val$F1)
        }
      }
      if (is.numeric(y)) {
        row <- data.frame(minobs = minobs, depth = depth,
                          MAE_train = mean(met_train$MAE, na.rm = TRUE),
                          MSE_train = mean(met_train$MSE, na.rm = TRUE),
                          MAPE_train = mean(met_train$MAPE, na.rm = TRUE),
                          MAE_val = mean(met_val$MAE, na.rm = TRUE),
                          MSE_val = mean(met_val$MSE, na.rm = TRUE),
                          MAPE_val = mean(met_val$MAPE, na.rm = TRUE))
        row$MAE_diff <- row$MAE_val - row$MAE_train
      } else {
        row <- data.frame(minobs = minobs, depth = depth,
                          Accuracy_train = mean(met_train$Accuracy, na.rm = TRUE),
                          Sensitivity_train = mean(met_train$Sensitivity, na.rm = TRUE),
                          Specificity_train = mean(met_train$Specificity, na.rm = TRUE),
                          F1_train = mean(met_train$F1, na.rm = TRUE),
                          Accuracy_val = mean(met_val$Accuracy, na.rm = TRUE),
                          Sensitivity_val = mean(met_val$Sensitivity, na.rm = TRUE),
                          Specificity_val = mean(met_val$Specificity, na.rm = TRUE),
                          F1_val = mean(met_val$F1, na.rm = TRUE))
        row$Accuracy_diff <- row$Accuracy_train - row$Accuracy_val
      }
      results[[length(results)+1]] <- row
    }
  }
  results_df <- do.call(rbind, results)
  return(results_df)
}

cv_tune_builtin_nn <- function(formula, data, sizes, decays, folds = 5, maxit = 5000) {
  library(caret)
  results <- list()
  mf <- model.frame(formula, data)
  y <- mf[[1]]
  folds_idx <- caret::createFolds(y, k = folds, list = TRUE)
  
  for (size in sizes) {
    for (decay in decays) {
      if (is.numeric(y)) {
        met_train <- list(MAE = c(), MSE = c(), MAPE = c())
        met_val   <- list(MAE = c(), MSE = c(), MAPE = c())
      } else {
        met_train <- list(Accuracy = c(), Sensitivity = c(), Specificity = c(), F1 = c())
        met_val   <- list(Accuracy = c(), Sensitivity = c(), Specificity = c(), F1 = c())
      }
      for (fold in folds_idx) {
        train_data <- data[-fold, ]
        val_data <- data[fold, ]
        if (is.numeric(y)) {
          model <- nnet(formula, data = train_data, size = size, decay = decay, linout = TRUE,
                        maxit = maxit, trace = FALSE)
          pred_train <- predict(model, newdata = train_data)
          pred_val   <- predict(model, newdata = val_data)
        } else {
          if (length(levels(y)) == 2) {
            model <- nnet(formula, data = train_data, size = size, decay = decay,
                          maxit = maxit, trace = FALSE)
            pred_train <- predict(model, newdata = train_data, type = "raw")
            pred_val   <- predict(model, newdata = val_data, type = "raw")
            if (is.matrix(pred_train) && ncol(pred_train) == 1) {
              pred_train <- as.numeric(pred_train)
            }
            if (is.matrix(pred_val) && ncol(pred_val) == 1) {
              pred_val <- as.numeric(pred_val)
            }
          } else {
            model <- nnet(formula, data = train_data, size = size, decay = decay,
                          maxit = maxit, trace = FALSE)
            pred_train <- predict(model, newdata = train_data, type = "raw")
            pred_val   <- predict(model, newdata = val_data, type = "raw")
            if (is.matrix(pred_train) && is.null(colnames(pred_train))) {
              colnames(pred_train) <- levels(y)
            }
            if (is.matrix(pred_val) && is.null(colnames(pred_val))) {
              colnames(pred_val) <- levels(y)
            }
          }
        }
        res_train <- modelOcena(train_data[[as.character(formula[[2]])]], pred_train)
        res_val   <- modelOcena(val_data[[as.character(formula[[2]])]], pred_val)
        if (is.numeric(y)) {
          met_train$MAE <- c(met_train$MAE, res_train$MAE)
          met_train$MSE <- c(met_train$MSE, res_train$MSE)
          met_train$MAPE <- c(met_train$MAPE, res_train$MAPE)
          met_val$MAE   <- c(met_val$MAE, res_val$MAE)
          met_val$MSE   <- c(met_val$MSE, res_val$MSE)
          met_val$MAPE  <- c(met_val$MAPE, res_val$MAPE)
        } else {
          met_train$Accuracy <- c(met_train$Accuracy, res_train$Accuracy)
          met_train$Sensitivity <- c(met_train$Sensitivity, res_train$Sensitivity)
          met_train$Specificity <- c(met_train$Specificity, res_train$Specificity)
          met_train$F1 <- c(met_train$F1, res_train$F1)
          met_val$Accuracy <- c(met_val$Accuracy, res_val$Accuracy)
          met_val$Sensitivity <- c(met_val$Sensitivity, res_val$Sensitivity)
          met_val$Specificity <- c(met_val$Specificity, res_val$Specificity)
          met_val$F1 <- c(met_val$F1, res_val$F1)
        }
      }
      if (is.numeric(y)) {
        row <- data.frame(size = size, decay = decay,
                          MAE_train = mean(met_train$MAE, na.rm = TRUE),
                          MSE_train = mean(met_train$MSE, na.rm = TRUE),
                          MAPE_train = mean(met_train$MAPE, na.rm = TRUE),
                          MAE_val = mean(met_val$MAE, na.rm = TRUE),
                          MSE_val = mean(met_val$MSE, na.rm = TRUE),
                          MAPE_val = mean(met_val$MAPE, na.rm = TRUE))
        row$MAE_diff <- row$MAE_val - row$MAE_train
      } else {
        row <- data.frame(size = size, decay = decay,
                          Accuracy_train = mean(met_train$Accuracy, na.rm = TRUE),
                          Sensitivity_train = mean(met_train$Sensitivity, na.rm = TRUE),
                          Specificity_train = mean(met_train$Specificity, na.rm = TRUE),
                          F1_train = mean(met_train$F1, na.rm = TRUE),
                          Accuracy_val = mean(met_val$Accuracy, na.rm = TRUE),
                          Sensitivity_val = mean(met_val$Sensitivity, na.rm = TRUE),
                          Specificity_val = mean(met_val$Specificity, na.rm = TRUE),
                          F1_val = mean(met_val$F1, na.rm = TRUE))
        row$Accuracy_diff <- row$Accuracy_train - row$Accuracy_val
      }
      results[[length(results)+1]] <- row
    }
  }
  results_df <- do.call(rbind, results)
  return(results_df)
}



CrossValidTune <- function(dane, kFold, parTune, seed, type, modelFun, predictFun, extraArgs = list()) {
  set.seed(seed)
  n <- nrow(dane)
  folds <- sample(rep(1:kFold, length.out = n))
  foldAssignments <- lapply(1:kFold, function(f) {
    assignment <- rep(1, n)
    assignment[which(folds == f)] <- 2
    return(assignment)
  })
  
  results_list <- list()
  
  for(i in 1:nrow(parTune)) {
    par_comb <- as.list(parTune[i, ])
    if(is.null(names(par_comb)) || any(names(par_comb) == "")) {
      names(par_comb) <- names(parTune)
    }
    
    metrics_train_list <- list()
    metrics_valid_list <- list()
    for(f in 1:kFold) {
      assignment <- foldAssignments[[f]]
      train_idx <- which(assignment == 1)
      valid_idx <- which(assignment == 2)
      train_data <- dane[train_idx, , drop = FALSE]
      valid_data <- dane[valid_idx, , drop = FALSE]
      response <- extraArgs$response
      predictors <- extraArgs$predictors
      trainX <- train_data[, predictors, drop = FALSE]
      trainY <- train_data[[response]]
      validX <- valid_data[, predictors, drop = FALSE]
      validY <- valid_data[[response]]
      
      model <- do.call(modelFun, c(list(trainX, trainY), par_comb))
      
      if(type %in% c("binary", "multiclass")) {
        pred_train <- predictFun(model, trainX, type = type)
        pred_valid <- predictFun(model, validX, type = type)
      } else {
        pred_train <- predictFun(model, trainX)
        pred_valid <- predictFun(model, validX)
      }
      
      eval_train <- modelOcena(trainY, pred_train)
      eval_valid <- modelOcena(validY, pred_valid)
      
      if(type == "regression") {
        m_train <- c(MAE = eval_train$MAE, RMSE = eval_train$RMSE, MAPE = eval_train$MAPE)
        m_valid <- c(MAE = eval_valid$MAE, RMSE = eval_valid$RMSE, MAPE = eval_valid$MAPE)
      } else if(type == "binary") {
        m_train <- c(Accuracy = eval_train$Accuracy, F1 = eval_train$F1, Youden = eval_train$YoudenIndex)
        m_valid <- c(Accuracy = eval_valid$Accuracy, F1 = eval_valid$F1, Youden = eval_valid$YoudenIndex)
      } else if(type == "multiclass") {
        m_train <- c(Accuracy = eval_train$OverallAccuracy, MacroF1 = eval_train$MacroF1, Recall = eval_train$MacroSensitivity)
        m_valid <- c(Accuracy = eval_valid$OverallAccuracy, MacroF1 = eval_valid$MacroF1, Recall = eval_valid$MacroSensitivity)
      }
      metrics_train_list[[f]] <- m_train
      metrics_valid_list[[f]] <- m_valid
    }
    avg_train <- Reduce("+", metrics_train_list) / kFold
    avg_valid <- Reduce("+", metrics_valid_list) / kFold
    if(type == "regression") {
      diff <- avg_valid - avg_train
    } else {
      diff <- avg_train - avg_valid
    }
    sum_diff <- sum(abs(diff))
    results_list[[i]] <- c(par_comb, avg_train, avg_valid, diff, SumDiff = sum_diff)
  }
  
  results_df <- do.call(rbind, lapply(results_list, function(x) as.data.frame(t(x), stringsAsFactors = FALSE)))
  rownames(results_df) <- NULL
  return(results_df)
}



# Wrapper
tuneKNN <- function(trainX, trainY, k) {
  model <- KNNtrain(trainX, trainY, k = k)
  return(model)
}
predictKNN <- function(model, newX, type = "regression") {
  preds <- KNNpredict(model, newX)
  return(preds)
}

# Drzewa Wrapper
tuneTree <- function(trainX, trainY, max_depth, minobs) {
  model_type <- if(is.numeric(trainY)) "regression" else "classification"
  model <- myDecisionTree(trainX, trainY, type = model_type, max_depth = max_depth, minobs = minobs)
  return(model)
}
predictTree <- function(model, newX, type = "regression") {
  preds <- predict_tree(model, newX)
  return(preds)
}

# Sieci Neuronowe Wrapper
tuneNN <- function(trainX, trainY, h, lr, iter, seed) {
  # Ustalamy typ modelu na podstawie trainY:
  model_type <- if (is.numeric(trainY)) {
    "regression"
  } else if (length(levels(trainY)) == 2) {
    "binary"
  } else {
    "multiclass"
  }
  
  if (model_type == "multiclass") {

    trainX <- model.matrix(~ . - 1, data = trainX)
    trainY <- as.factor(trainY)
  }
  
  if (is.character(h)) {
    h <- as.numeric(unlist(strsplit(h, split = "[, ]+")))
  } else if (is.list(h)) {
    h <- unlist(h)
  }
  h <- as.numeric(h)
  
  model <- trainNN2(as.matrix(trainX), trainY, type = model_type,
                    h = h, lr = lr, iter = iter, seed = seed)
  return(model)
}

predictNN <- function(model, newX, type = "regression") {
  if (type == "multiclass") {
    newX <- model.matrix(~ . - 1, data = newX)
  } else {
    newX <- as.matrix(newX)
  }
  return(predictNN2(newX, model, type = type))
}



plot_tuning_results_bar <- function(tuning_df, xvar, prefix) {
  library(reshape2)
  library(ggplot2)
  metric_cols <- grep(paste0("^", prefix, "_"), names(tuning_df), value = TRUE)
  plot_df <- melt(tuning_df, id.vars = xvar, measure.vars = metric_cols,
                  variable.name = "Metric", value.name = "Value")
  p <- ggplot(plot_df, aes_string(x = xvar, y = "Value", fill = "Metric")) +
    geom_bar(stat = "identity", position = "dodge") +
    labs(title = paste("Bar plot of", prefix, "metrics vs", xvar),
         x = xvar, y = "Value")
  return(p)
}

plot_tuning_results_line <- function(tuning_df, xvar, prefix) {
  library(reshape2)
  library(ggplot2)
  metric_cols <- grep(paste0("^", prefix, "_"), names(tuning_df), value = TRUE)
  plot_df <- melt(tuning_df, id.vars = xvar, measure.vars = metric_cols,
                  variable.name = "Metric", value.name = "Value")
  p <- ggplot(plot_df, aes_string(x = xvar, y = "Value", color = "Metric", group = "Metric")) +
    geom_line() + geom_point() +
    labs(title = paste("Line plot of", prefix, "metrics vs", xvar),
         x = xvar, y = "Value")
  return(p)
}


