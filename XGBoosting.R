#  ------------------------------------------------------------------------
#
#  Modelisation : 
#   - XGBoosting
#
#  ------------------------------------------------------------------------

# Package -----------------------------------------------------------------

library(data.table)
library(caret)
library(Ckmeans.1d.dp)
library(xgboost)


# Init --------------------------------------------------------------------

dataset[, Id := as.numeric(Id)]
setorder(dataset, Id)

my.RMSE = function(model, x, y, round = T, ...){
  pred_valid = predict(model, x, ...)
  if(isTRUE(round))
    pred_valid  = log(round(exp(pred_valid)/100, digits = 0)*100)
  sqrt(mean((pred_valid - y)**2))
}


id_outliers = c(524, 1299, 1183, 692)

set.seed(111)
idx = createDataPartition(1:1456, p = .85, list = F)

train = dataset[set == "train" & !Id %in% id_outliers, !colnames(dataset) %in% c("Id", "set", "logSalePrice"), with = F]

train.x = train[, colnames(train) != "SalePrice", with = F]
train.y = train[, colnames(train) == "SalePrice", with = F]

train_sub.x = train.x[idx,]
train_sub.y = train.y[idx,]

valid.x = train.x[-idx,]
valid.y = train.y[-idx,]

# Model XGBoosting-------------------------------------------------------------------

params <- list(
  "objective"           = "reg:linear",
  "eval_metric"         = "rmse"
)

X <- xgb.DMatrix(as.matrix(train_sub.x), label = train_sub.y$SalePrice)

resXGB <- xgboost(data = X, params = params, nrounds = 220) #was 60

my.RMSE(resXGB, as.matrix(train_sub.x), train_sub.y$SalePrice)
my.RMSE(resXGB, as.matrix(valid.x), valid.y$SalePrice)


importance <- xgb.importance(colnames(X), model = resXGB)
# install.packages("Ckmeans.1d.dp")
xgb.ggplot.importance(importance)

# make result based only on xgb
pred_xgb = predict(resXGB, as.matrix(dataset[set == "test", colnames(train.x), with = F]))

res = data.table(Id = dataset[set == "test"]$Id, SalePrice = pred_xgb)
setnames(res, c("Id", "SalePrice"))
# this Id has wrong prediction: you can see in pred_lasso
# thanks to proximity algorithm i looked houseprice around "same" observations
# res[Id == 2550, SalePrice := 269600]

write.csv(res, "submission.csv" ,row.names = F)

