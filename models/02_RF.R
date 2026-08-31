rm(list=ls())

library(ranger)
library(caret)

#### Datensätze laden ####

load(file = "export/2024/mz_24.RData")
load(file = "export/2024/azr_24.RData")
load(file = "export/2024/selten01_24.RData") # Vektor mit Staaten, deren Anteil Schutzsuchender <1%


# Train/Test Split --------------------------------------------------------

set.seed(358)

# random sample of 90% of the azr rows
trainid <- sample(x = 1:nrow(azr), size = 0.9*nrow(azr), replace=FALSE)

# training set
train_azr <- azr[trainid,]
# test set
test_azr_gesamt <- azr[-trainid,]

# remove countries where <1% are asylum seekers
train_azr<- train_azr[-which(train_azr$SA1 %in% selten01),] # ca. 5,443,000 übrig 

# split the test set
test_azr<-test_azr_gesamt[-which(test_azr_gesamt$SA1 %in% selten01),]
test_azr_selten<-test_azr_gesamt[which(test_azr_gesamt$SA1 %in% selten01),]


# Bernoulli fn ------------------------------------------------------------

# instead of using hard 0/1 predictions, use a probabilistic (Bernoulli) assignment
# outputs a probability p of seeking asylum;  a single Bernoulli draw (weighted by p) determines the final label
# capture prediction uncertainty 
BerSummary <- function (data, lev = NULL, model = NULL)  {
  if (length(levels(data$obs)) > 2)
    stop(paste("Your outcome has", length(levels(data$obs)),
               "levels. `BerSummary`` function isn't appropriate.",
               call. = FALSE))
  if (!all(levels(data[, "pred"]) == levels(data[, "obs"])))
    stop("Levels of observed and predicted data do not match.",
         call. = FALSE)
  
  # Schutzstatus zu ordnen
  for (i in 1:nrow(data)){
    # Binomialverteilung mit n=1 und p= vorhergesagte Wahrscheinlichkeit
    # für Schutzstatus=1
    pred_Status<- rbinom(n = 1, size = 1, prob = data$p[i])
    if(pred_Status==1){
      # in diesem Fall ist der Schutzstatus 1, wir tragen aber p ein, da 1 kein 
      # erlaubter Spaltenname ist
      data$pred[i]<-factor("p", levels=levels(data$obs))
    }
    else {
      # in diesem Fall ist der Schutzstatus 0, wir tragen aber n ein, da 0 kein 
      # erlaubter Spaltenname ist
      data$pred[i]<-factor("n", levels=levels(data$obs))
    }
  }
  # Precision und Recall werden berechnet
  c(Precision = caret::precision(data = data$pred, reference = data$obs, relevant = lev[1]),
    Recall = caret::recall(data = data$pred, reference = data$obs, relevant = lev[1]))
}


# Prepare training data ---------------------------------------------------
# for caret

# convert to data frame
# actually needed by caret::train()
azr_kreuz<-data.frame(train_azr)

# necessary for caret
levels(azr_kreuz$Schutzstatus) <- c("n", "p")

# reverse the factor level order to "p" is the positive class
# caret treats the first level as the positive class by default
azr_kreuz$Schutzstatus <- factor(azr_kreuz$Schutzstatus,
                                 levels=rev(levels(azr_kreuz$Schutzstatus)))


# Hyperparameter Tuning ---------------------------------------------------

# mtry: number of variables randomly sampled as candidates at each split
#       lower value increases diversity between trees -> more randomness
#       higher values make individual trees stronger but more correlated

# splitrule: creterion to choose the best split at each node
#            "gini" -> gini impurity
#            "extratrees" create extra randomness by randomizing split points -> improve performance on large datasets

# min.node.size: min number of observations recquired in the terminal node
#                large value = shallower, less overfit trees
#                smaller value = deeper, more flexible trees
grid<-expand.grid(
  mtry=c(2,3,4),
  splitrule=c("gini", "extratrees"),
  min.node.size=c(5,10)
)

# across cross-validation folds
set.seed(123)
seeds <- vector(mode = "list", length = 11)
for(i in 1:10) seeds[[i]] <- sample.int(1000, nrow(grid))
## For the last model:
seeds[[11]] <- sample.int(1000, 1)



# Cross-Validation Control -------------------------------------------------

# method: "cv" k-fold cross-validation
# number of folds
# classProbs: store prediction probabilities (needed by BerSummary and Bernoulli imputation)
# savePredictions: keep prediction for the best tuning parameter combination
# summaryFunction: use the custom function BerSummary
# seeds: for reproducibility

train_control<-trainControl(
  method="cv",
  number=10,
  classProbs=TRUE,
  savePredictions="final",
  summaryFunction=BerSummary,
  seeds=seeds
)



# CV with Parameter Tuning ------------------------------------------------

# training a random forest using ranger via caret, with all parameter combinations
# in a tuning grid with 10-fold CV
# maximize w.r.t Precision

# num.trees=200

model_rf_cv<-train(
  x=azr_kreuz[, 1:6],
  y=azr_kreuz$Schutzstatus,
  method="ranger",
  trControl=train_control,
  tuneGrid=grid,
  metric="Precision",
  num.trees=250,
  num.threads=4
)


# save model
save(model_rf_cv, file = "export/2024/model_rf_cv_24.rda")

print(model_rf_cv)
model_rf_cv$folds


# Training final model ----------------------------------------------------

# train a fresh random forest using the optimal hyperparameters from above
# on the full training dataset -> more data for prediction
best_mtry<-model_rf_cv$bestTune$mtry
best_splitrule<-model_rf_cv$bestTune$splitrule
best_min_node_size<-model_rf_cv$bestTune$min.node.size

# final model is trained directly with ranger (bypassing caret) using the original 0,1 labels
# probability=TRUE to obtain class probabilities rather than hard labels
model_final_rf<-ranger(
  formula=Schutzstatus~.,
  data=train_azr[, c(1:6, which(colnames(train_azr)=="Schutzstatus"))],
  num.trees=250,
  mtry=best_mtry,
  splitrule=best_splitrule,
  min.node.size=best_min_node_size,
  probability=TRUE,
  seed=358
)

save(model_final_rf, file="export/2024/model_final_rf_24.rda")


# Evaluation --------------------------------------------------------------

load(file="export/2024/model_final_rf_24.rda")

# predict class probabilities from non-rare countries
# ranger's predict() returns a list; $predictions is a matrix with one column per class
prob<-predict(object = model_final_rf, # verwendetes Modell
              newdata = test_azr[,1:6], 
              type = "response")$predictions

# for rare countries, do not use a model
# probability of 0 if seeking asylum and 1 if not seeking asylum
# -> asylum seeking from these groups are negligible
prob_selten<-cbind(rep(1,nrow(test_azr_selten)), # prob of "not seeking asylum"
                   rep(0,nrow(test_azr_selten))) # prob of "seeking asylum"

# combine model-predicted probabilities and rule-assigned proabilities
prob_gesamt<-rbind(prob,prob_selten)

# combine datasets together
test_azr<-rbind(test_azr, test_azr_selten)

# assign a predicted asylum status using Bernoulli sampling
# for each person, drawing once from the Bernoulli dist with success prob
# equal to the model's predicted probability of seeking proection
pred_Status<-numeric(nrow(prob_gesamt))

for (i in 1:nrow(prob_gesamt)){
  # Binomialverteilung mit n=1 und p= vorhergesagte Wahrscheinlichkeit
  # für schutzsuchend
  pred_Status[i]<-rbinom(n = 1, size = 1, prob = prob_gesamt[i,2])
}
pred_Status<-factor(pred_Status, levels=levels(test_azr$Schutzstatus))

# append the predicted labels
test_azr$pred_Schutzstatus<- pred_Status


# Confusion matrix --------------------------------------------------------

cm <- confusionMatrix(test_azr$pred_Schutzstatus,
                      reference = test_azr$Schutzstatus,
                      positive="1")

print(cm)


# Variable importance -----------------------------------------------------

importance(model_final_rf)


# Multiple imputation -----------------------------------------------------

mz_selten<-mz[which(mz$SA1 %in% selten01),]
mz<- mz[-which(mz$SA1 %in% selten01),]

prob_mz<-predict(object = model_final_rf, 
                 newdata = mz[,1:6], 
                 type = "response")$predictions

prob_mz_selten<-cbind(rep(1,nrow(mz_selten)),
                      rep(0,nrow(mz_selten)))

prob_mz_alle<-rbind(prob_mz,prob_mz_selten)

MI_mz<-rbind(mz, mz_selten)
