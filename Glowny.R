library(dplyr)
library(caret)
library(class)
library(rpart)
library(rpart.plot)
library(nnet)
library(data.tree)
library(ggplot2)
library(pROC)
library(cluster)
source("funkcje.R") 

cat("==== REGRESJA: Life Expectancy Data ====\n")
df <- read.csv("Life Expectancy Data.csv") %>% sample_n(1000)

for(col in colnames(df)) {
  if(is.numeric(df[[col]])) {
    df[[col]][is.na(df[[col]])] <- median(df[[col]], na.rm = TRUE)
  }
}

df$Status <- as.numeric(df$Status == "Developed")
df <- df %>% select(-Country, -Year)

set.seed(123)
trainIndex <- createDataPartition(df$Life.expectancy, p = 0.8, list = FALSE)
train <- df[trainIndex, ]
test <- df[-trainIndex, ]

cor_matrix <- cor(train %>% select(-Life.expectancy), use = "pairwise.complete.obs")
high_cor <- findCorrelation(cor_matrix, cutoff = 0.9)
if(length(high_cor) > 0){
  train <- train[, -high_cor]
  test <- test[, -high_cor]
}
predictor_columns <- setdiff(names(train), "Life.expectancy")
normalize_zscore <- function(x){ (x - mean(x, na.rm = TRUE)) / sd(x, na.rm = TRUE) }
train[predictor_columns] <- as.data.frame(lapply(train[predictor_columns], normalize_zscore))
test[predictor_columns]  <- as.data.frame(lapply(test[predictor_columns], normalize_zscore))


set.seed(123)
train_knn <- train[sample(nrow(train), min(200, nrow(train))), ]
test_knn  <- test[sample(nrow(test), min(200, nrow(test))), ]

cat(">> Tuning KNN (regresja) przy uzyciu CrossValidTune\n")
parTune_knn <- data.frame(k = seq(1,15,by = 2))
tuning_knn <- CrossValidTune(dane = train_knn, kFold = 5, parTune = parTune_knn, seed = 123,
                             type = "regression", modelFun = tuneKNN, predictFun = predictKNN,
                             extraArgs = list(response = "Life.expectancy", predictors = predictor_columns))
print(tuning_knn)
p_knn_line_train <- plot_tuning_results_line(tuning_knn, xvar = "k", prefix = "train")
p_knn_line_valid <- plot_tuning_results_line(tuning_knn, xvar = "k", prefix = "valid")
print(p_knn_line_train)
print(p_knn_line_valid)

best_knn_params <- tuning_knn[which.min(as.numeric(as.character(tuning_knn$SumDiff))), ]
cat("Najlepsze parametry KNN (custom):\n")
print(best_knn_params)
best_k <- as.numeric(best_knn_params$k)
knn_model <- KNNtrain(train_knn[predictor_columns], train_knn$Life.expectancy, k = best_k)
knn_preds <- KNNpredict(knn_model, test_knn[predictor_columns])
reg_ocena_knn <- modelOcena(test_knn$Life.expectancy, knn_preds)
print(reg_ocena_knn)

cat(">> Tuning Drzew Decyzyjnych (regresja) przy uzyciu CrossValidTune\n")
parTune_tree <- data.frame(max_depth = 1:5, minobs = rep(5,5))
tuning_tree <- CrossValidTune(dane = train, kFold = 5, parTune = parTune_tree, seed = 123,
                              type = "regression", modelFun = tuneTree, predictFun = predictTree,
                              extraArgs = list(response = "Life.expectancy", predictors = predictor_columns))
print(tuning_tree)
p_tree_line_train <- plot_tuning_results_line(tuning_tree, xvar = "max_depth", prefix = "train")
p_tree_line_valid <- plot_tuning_results_line(tuning_tree, xvar = "max_depth", prefix = "valid")
print(p_tree_line_train)
print(p_tree_line_valid)

best_tree_params <- tuning_tree[which.min(as.numeric(as.character(tuning_tree$SumDiff))), ]
cat("Najlepsze parametry drzewa (custom):\n")
print(best_tree_params)
best_depth <- as.numeric(best_tree_params$max_depth)
best_minobs <- as.numeric(best_tree_params$minobs)
my_tree_reg <- myDecisionTree(train[predictor_columns], train$Life.expectancy,
                              type = "regression", max_depth = best_depth, minobs = best_minobs)
tree_preds <- predict_tree(my_tree_reg, test[predictor_columns])
reg_ocena_tree <- modelOcena(test$Life.expectancy, tree_preds)
print(reg_ocena_tree)

cat(">> Tuning Sieci Neuronowych (regresja) przy uzyciu CrossValidTune\n")
param_grid <- data.frame(
  h = I(list(c(5,5), c(10,10), c(15,15), c(5,5), c(10,10), c(15,15))),
  lr = c(0.01, 0.01, 0.01, 0.005, 0.005, 0.005),
  iter = rep(5000, 6),
  seed = rep(123, 6)
)
set.seed(123)
folds <- caret::createFolds(train$Life.expectancy, k = 5, list = TRUE)
cv_results <- list()
for(i in 1:nrow(param_grid)) {
  params <- param_grid[i, ]
  train_MAE <- numeric(length(folds))
  train_RMSE <- numeric(length(folds))
  train_MAPE <- numeric(length(folds))
  valid_MAE <- numeric(length(folds))
  valid_RMSE <- numeric(length(folds))
  valid_MAPE <- numeric(length(folds))
  for(j in seq_along(folds)) {
    fold_idx <- folds[[j]]
    cv_train <- train[-fold_idx, , drop = FALSE]
    cv_valid <- train[fold_idx, , drop = FALSE]
    nn_model <- trainNN2(
      as.matrix(cv_train[predictor_columns]),
      as.matrix(cv_train$Life.expectancy),
      type = "regression",
      h = params$h[[1]],
      lr = params$lr,
      iter = params$iter,
      seed = params$seed
    )
    preds_train <- predictNN2(as.matrix(cv_train[predictor_columns]), nn_model, type = "regression")
    preds_valid <- predictNN2(as.matrix(cv_valid[predictor_columns]), nn_model, type = "regression")
    eval_train <- modelOcena(cv_train$Life.expectancy, preds_train)
    eval_valid <- modelOcena(cv_valid$Life.expectancy, preds_valid)
    train_MAE[j] <- eval_train$MAE
    train_RMSE[j] <- eval_train$RMSE
    train_MAPE[j] <- eval_train$MAPE
    valid_MAE[j] <- eval_valid$MAE
    valid_RMSE[j] <- eval_valid$RMSE
    valid_MAPE[j] <- eval_valid$MAPE
  }
  avg_train_MAE <- mean(train_MAE)
  avg_train_RMSE <- mean(train_RMSE)
  avg_train_MAPE <- mean(train_MAPE)
  avg_valid_MAE <- mean(valid_MAE)
  avg_valid_RMSE <- mean(valid_RMSE)
  avg_valid_MAPE <- mean(valid_MAPE)
  diff_MAE <- avg_valid_MAE - avg_train_MAE
  diff_RMSE <- avg_valid_RMSE - avg_train_RMSE
  diff_MAPE <- avg_valid_MAPE - avg_train_MAPE
  SumDiff <- abs(diff_MAE) + abs(diff_RMSE) + abs(diff_MAPE)
  cv_results[[i]] <- data.frame(
    h = paste(params$h[[1]], collapse = ","),
    lr = params$lr,
    iter = params$iter,
    seed = params$seed,
    train_MAE = avg_train_MAE,
    train_RMSE = avg_train_RMSE,
    train_MAPE = avg_train_MAPE,
    valid_MAE = avg_valid_MAE,
    valid_RMSE = avg_valid_RMSE,
    valid_MAPE = avg_valid_MAPE,
    diff_MAE = diff_MAE,
    diff_RMSE = diff_RMSE,
    diff_MAPE = diff_MAPE,
    SumDiff = SumDiff
  )
}
cv_results_df <- do.call(rbind, cv_results)
print(cv_results_df)
p_nn_line_train <- plot_tuning_results_line(cv_results_df, xvar = "lr", prefix = "train")
p_nn_line_valid <- plot_tuning_results_line(cv_results_df, xvar = "lr", prefix = "valid")
print(p_nn_line_train)
print(p_nn_line_valid)

cat(">> CV Tuning Built-in KNN (regresja) przy uzyciu cv_tune_builtin_knn\n")
cv_knn_reg <- cv_tune_builtin_knn(Life.expectancy ~ ., data = train, k_values = seq(1,15,by = 2), folds = 5)
print(cv_knn_reg)
best_k_builtin_reg <- cv_knn_reg$k[which.min(cv_knn_reg$MAE_diff)]
cat("Najlepsze k dla built-in KNN (regresja):", best_k_builtin_reg, "\n")

cat(">> CV Tuning Built-in Drzew Decyzyjnych (regresja) przy uzyciu cv_tune_builtin_tree\n")
cv_tree_reg <- cv_tune_builtin_tree(Life.expectancy ~ ., data = train, depths = 1:5, minobs_values = rep(5,5), folds = 5)
print(cv_tree_reg)
best_depth_reg <- cv_tree_reg$depth[which.min(cv_tree_reg$MAE_diff)]
best_minobs_reg <- cv_tree_reg$minobs[which.min(cv_tree_reg$MAE_diff)]
cat("Najlepsze parametry dla built-in drzewa (regresja): depth =", best_depth_reg, ", minobs =", best_minobs_reg, "\n")

cat(">> CV Tuning Built-in Sieci Neuronowych (regresja) przy uzyciu cv_tune_builtin_nn\n")
cv_nn_reg <- cv_tune_builtin_nn(Life.expectancy ~ ., data = train, sizes = c(5,10), decays = c(0.01,0.005), folds = 5, maxit = 5000)
print(cv_nn_reg)

cat("==== KLASYFIKACJA BINARNA: Banana Quality Data ====\n")
banana <- read.csv("banana_quality.csv") %>% sample_n(1000)
banana$Quality <- as.factor(banana$Quality)
banana <- na.omit(banana)
set.seed(123)
bananaIndex <- createDataPartition(banana$Quality, p = 0.8, list = FALSE)
banana_train <- banana[bananaIndex, ]
banana_test  <- banana[-bananaIndex, ]
banana_predictors <- setdiff(names(banana_train), "Quality")
banana_train[banana_predictors] <- as.data.frame(lapply(banana_train[banana_predictors], normalize_zscore))
banana_test[banana_predictors]  <- as.data.frame(lapply(banana_test[banana_predictors], normalize_zscore))

cat(">> Tuning KNN (klasyfikacja binarna) przy uzyciu CrossValidTune\n")
set.seed(123)
banana_train_knn <- banana_train[sample(nrow(banana_train), min(200, nrow(banana_train))), ]
banana_test_knn  <- banana_test[sample(nrow(banana_test), min(200, nrow(banana_test))), ]
parTune_knn_bin <- data.frame(k = seq(1,15,by = 2))
tuning_knn_bin <- CrossValidTune(dane = banana_train_knn, kFold = 5, parTune = parTune_knn_bin, seed = 123,
                                 type = "binary", modelFun = tuneKNN, predictFun = predictKNN,
                                 extraArgs = list(response = "Quality", predictors = banana_predictors))
print(tuning_knn_bin)
best_knn_bin_params <- tuning_knn_bin[which.min(as.numeric(as.character(tuning_knn_bin$SumDiff))), ]
cat("Najlepsze parametry KNN (custom, binarna):\n")
print(best_knn_bin_params)
best_k_bin <- as.numeric(best_knn_bin_params$k)
knn_model_bin <- KNNtrain(banana_train_knn[banana_predictors],
                          banana_train_knn$Quality, k = best_k_bin)
knn_preds_bin <- KNNpredict(knn_model_bin, banana_test_knn[banana_predictors])
bin_ocena_knn <- modelOcena(banana_test_knn$Quality, knn_preds_bin)
print(bin_ocena_knn)

cat(">> Tuning Drzew Decyzyjnych (klasyfikacja binarna) przy uzyciu CrossValidTune\n")
parTune_tree_bin <- data.frame(max_depth = 1:5, minobs = rep(5,5))
tuning_tree_bin <- CrossValidTune(dane = banana_train, kFold = 5, parTune = parTune_tree_bin, seed = 123,
                                  type = "binary", modelFun = tuneTree, predictFun = predictTree,
                                  extraArgs = list(response = "Quality", predictors = banana_predictors))
print(tuning_tree_bin)
best_tree_bin_params <- tuning_tree_bin[which.min(as.numeric(as.character(tuning_tree_bin$SumDiff))), ]
cat("Najlepsze parametry drzewa (custom, binarna):\n")
print(best_tree_bin_params)
best_depth_bin <- as.numeric(best_tree_bin_params$max_depth)
best_minobs_bin <- as.numeric(best_tree_bin_params$minobs)
my_tree_bin <- myDecisionTree(banana_train[banana_predictors], banana_train$Quality,
                              type = "classification", max_depth = best_depth_bin, minobs = best_minobs_bin)
tree_preds_bin <- predict_tree(my_tree_bin, banana_test[banana_predictors])
bin_ocena_tree <- modelOcena(banana_test$Quality, tree_preds_bin)
print(bin_ocena_tree)

cat(">> Tuning Sieci Neuronowych (klasyfikacja binarna) przy uzyciu CrossValidTune\n")
parTune_nn_bin <- data.frame(h = I(list(c(5,5), c(10,10))), lr = c(0.01, 0.005),
                             iter = c(5000,5000), seed = c(123,123))
tuning_nn_bin <- CrossValidTune(dane = banana_train, kFold = 5, parTune = parTune_nn_bin, seed = 123,
                                type = "binary", modelFun = tuneNN, predictFun = predictNN,
                                extraArgs = list(response = "Quality", predictors = banana_predictors))
print(tuning_nn_bin)


cat(">> CV Tuning Built-in KNN (klasyfikacja binarna) przy uzyciu cv_tune_builtin_knn\n")
cv_knn_bin <- cv_tune_builtin_knn(Quality ~ ., data = banana_train, k_values = seq(1,15,by=2), folds = 5)
print(cv_knn_bin)
best_k_builtin_bin <- cv_knn_bin$k[which.min(cv_knn_bin$Accuracy_diff)]
cat("Najlepsze k dla built-in KNN (binarna):", best_k_builtin_bin, "\n")

cat(">> CV Tuning Built-in Drzew Decyzyjnych (klasyfikacja binarna) przy uzyciu cv_tune_builtin_tree\n")
cv_tree_bin <- cv_tune_builtin_tree(Quality ~ ., data = banana_train, depths = 1:5, minobs_values = rep(5,5), folds = 5)
print(cv_tree_bin)
best_depth_bin2 <- cv_tree_bin$depth[which.min(cv_tree_bin$Accuracy_diff)]
best_minobs_bin2 <- cv_tree_bin$minobs[which.min(cv_tree_bin$Accuracy_diff)]
cat("Najlepsze parametry dla built-in drzewa (binarna): depth =", best_depth_bin2, ", minobs =", best_minobs_bin2, "\n")

cat(">> CV Tuning Built-in Sieci Neuronowych (klasyfikacja binarna) przy uzyciu cv_tune_builtin_nn\n")
cv_nn_bin <- cv_tune_builtin_nn(Quality ~ ., data = banana_train, sizes = c(5,10), decays = c(0.01,0.005), folds = 5, maxit = 5000)
print(cv_nn_bin)
best_size_bin2 <- cv_nn_bin$size[which.min(cv_nn_bin$Accuracy_diff)]
best_decay_bin2 <- cv_nn_bin$decay[which.min(cv_nn_bin$Accuracy_diff)]
cat("Najlepsze parametry dla built-in NN (binarna): size =", best_size_bin2, ", decay =", best_decay_bin2, "\n")

cat("==== KLASYFIKACJA WIELOKLASOWA: Game Engagement Data ====\n")
game <- read.csv("online_gaming_behavior_dataset.csv") %>% sample_n(1000)
game <- game %>% select(-PlayerID)
game$Gender <- as.factor(game$Gender)
game$Location <- as.factor(game$Location)
game$GameGenre <- as.factor(game$GameGenre)
game$GameDifficulty <- as.factor(game$GameDifficulty)
game$EngagementLevel <- as.factor(game$EngagementLevel)
game <- na.omit(game)
set.seed(123)
gameIndex <- createDataPartition(game$EngagementLevel, p = 0.8, list = FALSE)
game_train <- game[gameIndex, ]
game_test  <- game[-gameIndex, ]
game_predictors <- setdiff(names(game_train), "EngagementLevel")

game_train_num <- game_train
game_test_num <- game_test
for(var in game_predictors){
  if(is.factor(game_train_num[[var]])){
    game_train_num[[var]] <- as.numeric(game_train_num[[var]])
    game_test_num[[var]]  <- as.numeric(game_test_num[[var]])
  }
}


cat(">> Tuning KNN (wieloklasowa) przy uzyciu CrossValidTune\n")
set.seed(123)
game_train_knn <- game_train[sample(nrow(game_train), min(200, nrow(game_train))), ]
game_test_knn  <- game_test[sample(nrow(game_test), min(200, nrow(game_test))), ]
parTune_knn_multi <- data.frame(k = seq(1,15,by = 2))
tuning_knn_multi <- CrossValidTune(dane = game_train_knn, kFold = 5, parTune = parTune_knn_multi, seed = 123,
                                   type = "multiclass", modelFun = tuneKNN, predictFun = predictKNN,
                                   extraArgs = list(response = "EngagementLevel", predictors = game_predictors))
print(tuning_knn_multi)
best_knn_multi_params <- tuning_knn_multi[which.min(as.numeric(as.character(tuning_knn_multi$SumDiff))), ]
cat("Najlepsze parametry KNN (custom, wieloklasowa):\n")
print(best_knn_multi_params)
best_k_multi <- as.numeric(best_knn_multi_params$k)
knn_model_multi <- KNNtrain(game_train_knn[game_predictors], game_train_knn$EngagementLevel, k = best_k_multi)
knn_preds_multi <- KNNpredict(knn_model_multi, game_test_knn[game_predictors])
multi_ocena_knn <- modelOcena(game_test_knn$EngagementLevel, knn_preds_multi)
print(multi_ocena_knn)

cat(">> Tuning Drzew Decyzyjnych (wieloklasowa) przy uzyciu CrossValidTune\n")
parTune_tree_multi <- data.frame(max_depth = 1:5, minobs = rep(5,5))
tuning_tree_multi <- CrossValidTune(dane = game_train_num, kFold = 5, parTune = parTune_tree_multi, seed = 123,
                                    type = "multiclass", modelFun = tuneTree, predictFun = predictTree,
                                    extraArgs = list(response = "EngagementLevel", predictors = game_predictors))
print(tuning_tree_multi)
best_tree_multi_params <- tuning_tree_multi[which.min(as.numeric(as.character(tuning_tree_multi$SumDiff))), ]
cat("Najlepsze parametry drzewa (custom, wieloklasowa):\n")
print(best_tree_multi_params)
best_depth_multi <- as.numeric(best_tree_multi_params$max_depth)
best_minobs_multi <- as.numeric(best_tree_multi_params$minobs)
my_tree_multi <- myDecisionTree(game_train_num[game_predictors],
                                game_train_num$EngagementLevel,
                                type = "classification", max_depth = best_depth_multi, minobs = best_minobs_multi)
tree_preds_multi <- predict_tree(my_tree_multi, game_test_num[game_predictors])
multi_ocena_tree <- modelOcena(game_test_num$EngagementLevel, tree_preds_multi)
print(multi_ocena_tree)

cat(">> Tuning Sieci Neuronowych (wieloklasowa) przy uzyciu CrossValidTune\n")
parTune_nn_multi <- data.frame(h = I(list(c(10,10), c(15,15))), lr = c(0.01,0.005),
                               iter = c(5000,5000), seed = c(123,123))
tuning_nn_multi <- CrossValidTune(dane = game_train, kFold = 5, parTune = parTune_nn_multi, seed = 123,
                                  type = "multiclass", modelFun = tuneNN, predictFun = predictNN,
                                  extraArgs = list(response = "EngagementLevel", predictors = game_predictors))
print(tuning_nn_multi)
best_nn_multi_params <- tuning_nn_multi[which.min(as.numeric(as.character(tuning_nn_multi$SumDiff))), ]
cat("Najlepsze parametry NN (custom, wieloklasowa):\n")
print(best_nn_multi_params)


cat(">> CV Tuning Built-in KNN (wieloklasowa) przy uzyciu cv_tune_builtin_knn\n")
cv_knn_multi <- cv_tune_builtin_knn(EngagementLevel ~ ., data = game_train, k_values = seq(1,15,by=2), folds = 5)
print(cv_knn_multi)
best_k_builtin_multi <- cv_knn_multi$k[which.min(cv_knn_multi$Accuracy_diff)]
cat("Najlepsze k dla built-in KNN (wieloklasowa):", best_k_builtin_multi, "\n")

cat(">> CV Tuning Built-in Drzew Decyzyjnych (wieloklasowa) przy uzyciu cv_tune_builtin_tree\n")
cv_tree_multi <- cv_tune_builtin_tree(EngagementLevel ~ ., data = game_train, depths = 1:5, minobs_values = rep(5,5), folds = 5)
print(cv_tree_multi)
best_depth_multi2 <- cv_tree_multi$depth[which.min(cv_tree_multi$Accuracy_diff)]
best_minobs_multi2 <- cv_tree_multi$minobs[which.min(cv_tree_multi$Accuracy_diff)]
cat("Najlepsze parametry dla built-in drzewa (wieloklasowa): depth =", best_depth_multi2, ", minobs =", best_minobs_multi2, "\n")

cat(">> CV Tuning Built-in Sieci Neuronowych (wieloklasowa) przy uzyciu cv_tune_builtin_nn\n")
cv_nn_multi <- cv_tune_builtin_nn(EngagementLevel ~ ., data = game_train, sizes = c(10,15), decays = c(0.01,0.005), folds = 5, maxit = 5000)
print(cv_nn_multi)
best_size_multi2 <- cv_nn_multi$size[which.min(cv_nn_multi$Accuracy_diff)]
best_decay_multi2 <- cv_nn_multi$decay[which.min(cv_nn_multi$Accuracy_diff)]
cat("Najlepsze parametry dla built-in NN (wieloklasowa): size =", best_size_multi2, ", decay =", best_decay_multi2, "\n")


