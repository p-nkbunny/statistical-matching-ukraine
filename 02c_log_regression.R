# logistic regression

rm(list=ls())

library(data.table)
library(MASS)
library(parallel)

options(datatable.optimize = 2)

# loading datasets
load(file = "export/2024/mz_24.RData") # recipient
load(file = "export/2024/azr_24.RData") # donor

setDT(azr)
setDT(mz)

# remove unneeded columns
# azr[, c("lon", "lat") := NULL]



# model specification -------------------------------------------------------------------
#

# specify the regression formula here
# nest kreis within Bundesland
# default formula for the following sections
# get_hierarchical_fn<-function() {
#   as.formula("Schutzstatus ~ Geschlecht + Alter + JahrZuzug + SA1 + Verheiratet + Bundesland + kreis")
# }


# polynomial spatial surface
formula_polynomial_spatial<-function() {
  formula<-as.formula("Schutzstatus ~ Geschlecht + Alter + JahrZuzug + SA1 + Verheiratet + Bundesland +
             lon + lat + I(lon^2) + I(lat^2) + lon:lat")
  return(formula)
}




# model validation fn -----------------------------------------------------

check_model_validity<-function(model, model_name="Model_123") {
  
  # initialize variables
  all_checks_passed<-TRUE
  warnings_list<-character(0)
  errors_list<-character(0)
  
  # 1: Convergence
  cat("Checking convergence\n")
  if (!is.null(model$converged)) {
    if (model$converged) {
      cat("model converged :)\n")
    }
    else {
      cat("model did not converge :(\n")
      warnings_list<-c(warnings_list, "model did not converge")
      all_checks_passed<-FALSE
    }
  }
  else {
    cat("convergence status not available\n")
  }
  
  # 2: boundary (complete or quasi-complete separation)
  cat("Checking boundary (separation) issues\n")
  if (!is.null(model$boundary)) {
    if (model$boundary) {
      cat("boundary (complete or quasi-complete separation) detected\n")
      cat("some predictors perfectly predict the outcome\n")
      warnings_list<-c(warnings_list, "boundary/separation detected")
      all_checks_passed<-FALSE
    }
    else {
      cat("no boundary issues :)\n")
    }
  }
  
  # 3: check for extreme values in the coefficients
  cat("Checking coefficient magnitues\n")
  coefs<-coef(model)
  extreme_coefs<-abs(coefs)>10
  
  if (any(extreme_coefs, na.rm=TRUE)) {
    cat("some coefficients have extreme values (abs(Beta)>10)\n")
    extreme_names<-names(coefs)[which(extreme_coefs)]
    for (name in extreme_names[1:min(5, length(extreme_names))]) {
      cat(sprintf("%s %.2f\n", name, coefs[name]))
    }
    
    cat("maybe there is separation or collinearity\n")
    warnings_list<-c(warnings_list, "extreme coefficient values")
  }
  else {
    cat("all coefficients are within reasonable range\n")
  }
  
  # 4: NA coefficients
  cat("Checking for missing coefficients\n")
  na_coefs<-sum(is.na(coefs))
  if (na_coefs>0) {
    cat(sprintf("%d coeffs are NA\n", na_coefs))
    caterrors_list<-c(errors_list, "NA coeffs are present")
    all_checks_passed<-FALSE
  }
  else {
    cat("no NA coefficients :)\n")
  }
  
  # 5: standard errors
  cat("Checking standard errors\n")
  se<-sqrt(diag(vcov(model)))
  extreme_se<-se>100
  
  if (any(extreme_se, na.rm=TRUE)) {
    cat("some standard errors are large -> unstable parameter estimates\n")
    warnings_list<-c(warnings_list, "Large standard errors")
  }
  else {
    cat("standard errosr are within an acceptable range\n")
  }
  
  # 6: Fitted probabilities
  cat("Checking fitted probabilities\n")
  fitted_probs<-fitted(model)
  extreme_probs<-fitted_probs<0.01 | fitted_probs>0.99
  pct_extreme<-100*mean(extreme_probs)
  
  cat(sprintf("%.1f%% fitted probs are near 0 or 1\n", pct_extreme))
  if (pct_extreme>50) {
    cat("many extreme probabilities -> overfitting\n")
    warnings_list<-c(warnings_list, "many extreme probabilities")
  }
  else if (pct_extreme>20) {
    cat("some extreme probabilities present\n")
  }
  else {
    cat("reasonable fitted probabilities\n")
  }
  
  # summary
  if (all_checks_passed && length(warnings_list)==0) {
    cat("all checks passed\n")
  }
  else if (length(errors_list)>0) {
    cat("critical errors detected\n")
  }
  else if(length(warnings_list)>0) {
    cat("warnings present\n")
  }
  
  return(list(
    valid=all_checks_passed && length(errors_list)==0,
    warnings=warnings_list,
    errors=errors_list,
    converged=isTRUE(model$converged),
    pct_extreme_probs=pct_extreme
  ))
}




# multiple imputation -----------------------------------------------------
# running logistic imputation multiple times

pre_validation_check<-function(donor, recipient, n_imputations = 10,
                            formula=NULL, validate_model=TRUE, test_proportion=0.2) {
  
  # initialize variables
  validation_results<-NULL
  
  if (validate_model) {
    cat("-----Validating on Hold-out test set-------\n")
    n_test<-ceiling(nrow(donor)*test_proportion)
    test_idx<-sample(nrow(donor), n_test)
    
    donor_train<-donor[-test_idx]
    donor_test<-donor[test_idx]
    cat(sprintf("training set %d observations\n", nrow(donor_train)))
    
    cat("fitting on training set \n")
    model_val<-glm(formula, data=donor_train, family=binomial())
    cat("model fit\n")
    
    validity<-check_model_validity(model_val, "Validation Model")
    
    if (!validity$valid) {
      warning("model validation has critical issues")
    }
    
    rm(model_val, donor_train, donor_test)
    gc()
  }
  
}  

fitting_logreg_model<-function(donor, formula=NULL) {
  
  # fitting the model on the entire donor dataset
  cat("Step 1: fitting model once on full donor set\n")
  start_time<-Sys.time()
  model<-glm(formula, data=donor, family=binomial(link="logit"))
  
  fit_time<-difftime(Sys.time(), start_time, unit="secs")
  cat(sprintf("model took %.2f seconds\n", fit_time))
  cat(sprintf("number of parameters: %d\n", length(coef(model))))
  
  return(model)
  
}


logreg_validity<-function(model) {
  
  # checking model validity
  validity_check<-check_model_validity(model, "final model")
  
  if (!validity_check$valid) {
    stop("model has critical errors :O\n")
  }
  
  return(validity_check)
}

logreg_regression<-function(model, formula, recipient, n_imputations=10) {
  
  # extract parameters and covariance
  beta_hat<-coef(model)
  V<-vcov(model)
  
  # create design matrix for the recipient data
  formula_rhs<-formula[-2] # removing response variable
  
  # saving the number of rows of the recipient
  n_recipient_original<-nrow(recipient)
  
  recipient_indexed<-copy(recipient)
  recipient_indexed[, .row_index:=.I]
  
  tryCatch({
    X_recipient<-model.matrix(formula_rhs, data=recipient_indexed)
    
    # get row indices that were kept
    # needed if rows were dropped due to NAs
    kept_indices<-as.numeric(rownames(X_recipient))
    
    cat(sprintf("%d rows x %d columns\n", nrow(X_recipient), ncol(X_recipient)))
    
    if (nrow(X_recipient)<n_recipient_original) {
      
      n_dropped<-n_recipient_original-nrow(X_recipient)
      cat(sprintf("%d rows dropped\n", n_dropped))
      
      
    }
    
    if (ncol(X_recipient) != length(beta_hat)) {
      stop("design matrix dim do not match coefficients\n")
    }
  }, error = function(e) {
    stop("error creating design matrix: ", e$message)
  })
  
  # generate imputations by drawing from posterior
  
  imputed_data<-as.data.table(recipient)
  
  for (m in 1:n_imputations) {
    cat(sprintf("imputation %d\n", m))
    
    # draw parameters from posterior distribution
    beta_m <- mvrnorm(1, mu=beta_hat, Sigma=V)
    
    # compute predicted probabilities
    cat("computing predicted probs..\n")
    log_odds<-X_recipient %*% beta_m
    pred_probs<-plogis(log_odds)
    
    # draw binary outcomes
    set.seed(123+m)
    imputed_status<-rbinom(nrow(recipient), size=1, prob=pred_probs)
    
    # store
    col_name<-paste0("Schutz_", m)
    
    full_imputed<-rep(NA_integer_, nrow(imputed_data))
    full_imputed[kept_indices]<-imputed_status
    imputed_data[, (col_name):=full_imputed]
    
    cat(sprintf("mean refugee rate: %.3f\n", mean(imputed_status)))
    
    # store probs from first imputation
    if (m==1) {
      imputed_data[, prob_refugee:=NA_real_]
      imputed_data[kept_indices, prob_refugee:=as.vector(pred_probs)]
    }
  }
  
  
  rm(model, beta_hat, V, X_recipient)
  gc()
  
  return(
    data=imputed_data
  )
  
  
  
}

# running -----------------------------------------------------------------

# have to shift SA1 column as missings are causing issues
# names(mz)[names(mz)=='SA1']<-'SA1_azr'
# names(mz)[names(mz)=='SA1_preserved']<-'SA1'

formula<-formula_polynomial_spatial()

# fitting the model
logreg_mi<-fitting_logreg_model(donor=azr, formula=formula)

# checking validity
logreg_validity(logreg_mi)

# perform multiple imputation
results<-logreg_regression(model=logreg_mi, formula=formula, recipient=mz)


save(logreg_mi, file='export/2024/logreg_polynomial_mi.RData')
load('export/2024/logreg_polynomial_mi.RData')
