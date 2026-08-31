#### C5.0 Decision Trees ####
# The file contains:
# the split into training and test data,
# parameter optimization using precision, 
# training of the final model,
# evaluation of the final model using the test data, and
# multiple imputation in the microcensus

rm(list=ls())

library(C50)    # für decision trees (C5.0)
library(caret)  # zur Evaluation (Konfusionsmatrix, Kreuzvalidierung)
library(base)   # um expand.grid() bei der Kreuzvalidierung zu verwenden

#### Load datasets ####

load(file = "export/2024/mz_24.RData")
load(file = "export/2024/azr_24.RData")
load(file = "export/2024/selten01_24.RData") # countries where refugee seekers make up <1% of the total

#### ************************ Trainings- und Testdaten ******************* ####

set.seed(358)

# A number equal to (0.9 * the number of rows in the azr) is drawn from the set of numbers 1, ..., the number of rows 
# in the AZR (without replacement)
trainid <- sample(x = 1:nrow(azr), size = 0.9*nrow(azr), replace=FALSE)

# Training data consists of the data whose row numbers were previously selected
train_azr <- azr[trainid,]
# Testdaten
test_azr_gesamt <- azr[-trainid,]

### removing selton01 from the complete list
train_azr<- train_azr[-which(train_azr$SA1 %in% selten01),] # ca. 5,443,000 übrig 

### Split test data (removing from selton01)
test_azr<-test_azr_gesamt[-which(test_azr_gesamt$SA1 %in% selten01),]
test_azr_selten<-test_azr_gesamt[which(test_azr_gesamt$SA1 %in% selten01),]



#### ********* Parameter Optimization Using Cross-Validation and Precision ******** ####
# Optimizing on precision when predicting refugee status

#### Custom summary function for Precision ####
# # Refugee status more likely to be assigned to the class
# 
# # Source: https://github.com/topepo/caret/blob/master/pkg/caret/R/postResample.R
# # The MLmetrics package is required for prSummary. The function has been modified
# # to use the existing caret package. However, PR AUC cannot be determined with this method.
# prSummary2 <- function (data, lev = NULL, model = NULL)  {
#   
#   if (length(levels(data$obs)) > 2)
#     stop(paste("Your outcome has", length(levels(data$obs)),
#                "levels. `prSummary`` function isn't appropriate.",
#                call. = FALSE))
#   if (!all(levels(data[, "pred"]) == levels(data[, "obs"])))
#     stop("Levels of observed and predicted data do not match.",
#          call. = FALSE)
#   # der Schutzstatus wird der Klasse zugeordnet, die die größere Wahrscheinlichkeit hat
#   # Gemäß dieser Zuordnung werden Precision und Recall berechnet.
#   c(Precision = caret::precision(data = data$pred, reference = data$obs, relevant = lev[1]),
#     Recall = caret::recall(data = data$pred, reference = data$obs, relevant = lev[1]))
# }
# 

#### Custom summaryFunction with a Bernoulli trial ####
# Since we want to assign the protection status using the Bernoulli distribution,
# we need a custom function for this

# assign protection status using Bernoulli distribution
BerSummary <- function (data, lev = NULL, model = NULL)  {
  if (length(levels(data$obs)) > 2)
    stop(paste("Your outcome has", length(levels(data$obs)),
               "levels. `BerSummary`` function isn't appropriate.",
               call. = FALSE))
  if (!all(levels(data[, "pred"]) == levels(data[, "obs"])))
    stop("Levels of observed and predicted data do not match.",
         call. = FALSE)
  
  # Assigning refugee status
  for (i in 1:nrow(data)){
    pred_Status<- rbinom(n = 1, size = 1, prob = data$p[i])
    if(pred_Status==1){
      # in the case refugee seeking status is 1, but "p" is entered
      # as "1" cannot be a valid column name
      data$pred[i]<-factor("p", levels=levels(data$obs))
    }
    else {
      # same case as above, except status is 0 and "n" is the column name
      data$pred[i]<-factor("n", levels=levels(data$obs))
    }
  }
  # calculating Precision and Recall
  c(Precision = caret::precision(data = data$pred, reference = data$obs, relevant = lev[1]),
    Recall = caret::recall(data = data$pred, reference = data$obs, relevant = lev[1]))
}


#### Optimization with Precision ####
# Converted to a dataframe to avoid error messages
azr_kreuz<-data.frame(train_azr)

# Labels 0->n, 1->p are changed to avoid errors
# since 0 and 1 are not valid column names
levels(azr_kreuz$Schutzstatus) <- c("n", "p")

# Swap the order of the factor's levels (this does not change the refugee seeking status),
# because for "train," the first level is the positive class
azr_kreuz$Schutzstatus <- factor(azr_kreuz$Schutzstatus,
                                 levels=rev(levels(azr_kreuz$Schutzstatus))) # rev() vertauscht die Reihenfolge



# C5.0 has the following Tuning Parameters:
# trials (integer), model (tree, rules), winnow (TRUE, FALSE)

# Data frame containing the parameters to be tested
# `expand.grid` creates a data frame containing all combinations of the specified vectors
grid <-  expand.grid(trials =c(1, 2, 3, 4),
                     model = c('tree'),
                     winnow= c(FALSE))

set.seed(123)
seeds <- vector(mode = "list", length = 11)
for(i in 1:10) seeds[[i]] <- sample.int(1000, nrow(grid))
## For the last model:
seeds[[11]] <- sample.int(1000, 1)

# Function that sets some cross-validation parameters
train_control <- trainControl(method = "cv", # Kreuzvalidierung
                              number = 10,   # Anzahl der Teilmengen
                              classProbs = TRUE, # Wahrscheinlichkeiten werden gespeichert
                              savePredictions = 'final', # saves predictions for optimal tuning parameter
                              summaryFunction = BerSummary, # Verwendung der eigenen Metrik // use own metrics
                              seeds=seeds) # seeds setzen !!! wurde noch nicht ausprobiert !!!


# Cross/validation with parameter tuning
model_prec_bern<-train(x=azr_kreuz[,1:6], y=azr_kreuz$Schutzstatus,
                       method="C5.0", trControl = train_control,
                       tuneGrid = grid,
                       metric="Precision")

# Saving model
save(model_prec_bern, file = "export/2024/model_prec_bern_24_2.rda" )

# Ergebnisse der Kreuzvalidierung ausgeben
print(model_prec_bern)
# Ergebnisse mit sd ausgeben
model_prec_bern$results

#### ************************** Training final model ******************* ####
# can skip this if you already have a final model 


# Using the Optimal Parameters for Training
model_final<- C5.0(x=train_azr[,1:6], # Trainingsdaten
                   y = train_azr$Schutzstatus, # Schutzstatus
                   trials = 3, # Anzahl der Trials
                   control = C5.0Control(CF = 0.25))

# When rules=TRUE is specified, rules are created instead of the tree


## Modell speichern
save(model_final, file = "export/2024/model_final_24_2.rda")

# summary(model_final)

# Alternatively, using the model from the `train` function, you can obtain:
# model_prec_bern$finalModel
# The final model with the optimal parameters based on
# cross-validation
# In this case, the columns 
# containing the probabilities must be swapped in the next section, since the first column of the
# output from `predict()` indicates the probability for class 1 (seeking protection)


#### *************** Auswertung finales Modell mit Testdaten ************** ####
# evaluating final model with test data

## Load the model, unless it needs to be recreated
load(file="model_final_21_2.rda")

### Predicting Refugee-seeking Status
# Matrix of probabilities for conservation status (0 and 1)
prob<-predict(object = model_final, # verwendetes Modell
              newdata = test_azr[,1:6], 
              type = "prob") # Ausgabe der Wahrscheinlichkeiten

# Create a matrix of probabilities for countries where the proportion of asylum seekers is <1%
# Assign a probability of 0 for "asylum seeker" and 1 for "not an asylum seeker."
# (Depending on the training, make sure to note which of the two columns represents the probability 
# of being an asylum seeker)
prob_selten<-cbind(rep(1,nrow(test_azr_selten)),rep(0,nrow(test_azr_selten)))

# Matrices with probabilities listed
prob_gesamt<-rbind(prob,prob_selten)

# Combine test data (row bound)
test_azr<-rbind(test_azr, test_azr_selten)

# empty vector for the predicted value
pred_Status<-numeric(nrow(prob_gesamt))

# Determine predicted values
for (i in 1:nrow(prob_gesamt)){
  # Binomial distribution with n=1 and p= predicted probability
  # for those seeking protection
  pred_Status[i]<-rbinom(n = 1, size = 1, prob = prob_gesamt[i,2])
}
pred_Status<-factor(pred_Status, levels=levels(test_azr$Schutzstatus))

# Predicted values as a new column in the test data
test_azr$pred_Schutzstatus<- pred_Status


#### Confusion matrix ####
cm <- confusionMatrix(test_azr$pred_Schutzstatus,
                      reference = test_azr$Schutzstatus,
                      positive="1")
print(cm)

#### Attribute Usage ####
summary(model_final)

# Percentage of training data for which the variable was involved in 
# assigning the data to a leaf of the tree
C5imp(model_final, metric = "usage")
# Percentage of splits associated with this variable 
C5imp(model_final, metric = "splits")


#### Tree saved as txt  ####
# write(capture.output(summary(mod_dec_tree_3_neu)), "c50_Tree.txt")




#### ************************* MI im mz *********************************** ####


mz_selten<-mz[which(mz$SA1 %in% selten01),]
mz<- mz[-which(mz$SA1 %in% selten01),] #  ca. 40,000 übrig

# Calculating Probabilities
prob_mz<-predict(object = model_final, newdata = mz[,1:6], type = "prob")

# For all countries where the proportion of people seeking protection is less than 1%, the Wskt value is set to 0 for those seeking protection
# and 1 for those not seeking protection
prob_mz_selten<-cbind(rep(1,nrow(mz_selten)),rep(0,nrow(mz_selten)))

# Merging Probability Matrices
prob_mz_alle<-rbind(prob_mz,prob_mz_selten)

# creates a new dataset with 10 additional columns (with imputed values)
MI_mz<-rbind(mz, mz_selten)

# Utility function for identifying a column with imputed values
imp_function <- function(){
  zeile<-0
  # Vector in which values are stored
  pred_Schutzstatus<-numeric(nrow(MI_mz))
  for (i in 1:nrow(MI_mz)){
    zeile<-zeile+1
    # Probability in each row
    wahrscheinlichkeit<- prob_mz_alle[i,2]
    # Projected refugee seeking status
    pred_Schutzstatus[zeile]<- rbinom(n = 1, size = 1, prob =wahrscheinlichkeit)
    
  }
  return(pred_Schutzstatus)
}

# Add 10 columns with imputed values to the dataset
MI_mz$Schutz_1 <- factor(x = imp_function(), levels = levels(test_azr$Schutzstatus))
MI_mz$Schutz_2 <- factor(x = imp_function(), levels = levels(test_azr$Schutzstatus))
MI_mz$Schutz_3 <- factor(x = imp_function(), levels = levels(test_azr$Schutzstatus))
MI_mz$Schutz_4 <- factor(x = imp_function(), levels = levels(test_azr$Schutzstatus))
MI_mz$Schutz_5 <- factor(x = imp_function(), levels = levels(test_azr$Schutzstatus))
MI_mz$Schutz_6 <- factor(x = imp_function(), levels = levels(test_azr$Schutzstatus))
MI_mz$Schutz_7 <- factor(x = imp_function(), levels = levels(test_azr$Schutzstatus))
MI_mz$Schutz_8 <- factor(x = imp_function(), levels = levels(test_azr$Schutzstatus))
MI_mz$Schutz_9 <- factor(x = imp_function(), levels = levels(test_azr$Schutzstatus))
MI_mz$Schutz_10 <- factor(x = imp_function(), levels = levels(test_azr$Schutzstatus))


#### MI speichern ####
save(MI_mz, file='export/2024/MI_mz_24.RData')



