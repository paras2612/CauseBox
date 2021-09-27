install.packages("rJava",dependencies=TRUE,repos = "http://cran.us.r-project.org")
install.packages("grf",repos = "http://cran.us.r-project.org")
install.packages("rmatio",repos = "http://cran.us.r-project.org")
options(java.parameters = "-Xmx2500m",repos = "http://cran.us.r-project.org")
install.packages('Rcpp')
library(Rcpp)
library(grf)
library(ggplot2)
library(rmatio)
library(abind)
library(reticulate)


args <- commandArgs(trailingOnly = TRUE)
fn <- args[1]
test_fn <- args[2]
write_fn <- args[3]

dataset <- args[4]
model = "causal forest"
i_exps <- c(1:as.integer(args[5]))

trs <- c(0.6)


set.seed(42)

for (tr in trs){
  df = data.frame()
  for (i_exp in i_exps) {
    np<-import("numpy")
    npz1<-np$load(fn)
    npz1_test<-np$load(test_fn)
    
    X<-scale(npz1['x'][,,i_exp])
    X_test<-scale(npz1_test['x'][,,i_exp])
    n = dim(X)[1]
    T_test<-npz1_test['t'][,i_exp]
    YF_test<-npz1_test['yf'][,i_exp]
    XT_test <- abind(X_test,T_test,along=2)
    XT_CF_test <- abind(X_test,1-T_test,along=2)
    T<-npz1['t'][,i_exp]
    YF<-npz1['yf'][,i_exp]
    
    
    
    XT <- abind(X,T,along=2)
    XT_CF <- abind(X,1-T,along=2)
    
    forest.Y <- regression_forest(XT, YF)
    Y.hat <- predict(forest.Y)$predictions
    
    forest.Y_CF <- regression_forest(XT_CF, YF)
    Y_CF.hat <- predict(forest.Y_CF)$predictions
    
    forest.Y_test <- regression_forest(XT_test, YF_test)
    Y_test.hat <- predict(forest.Y_test)$predictions
    
    forest.Y_CF_test <- regression_forest(XT_CF_test, YF_test)
    YCF_test.hat <- predict(forest.Y_CF_test)$predictions
    
    c.forest = causal_forest(X, YF, T, Y.hat = Y.hat, num.trees = 10)
    
    #tau_i predicted for the test set
    c.pred = as.numeric(unlist(predict(c.forest, X_test)))
    
    c.cf.forest = causal_forest(X, YF, 1-T, Y.hat = Y_CF.hat, num.trees = 10)
    
    #tau_i predicted for the test set
    c.cf.pred = as.numeric(unlist(predict(c.cf.forest, X_test)))
    
    
    rmse_fact = sqrt(mean((Y_test.hat - YF_test)**2))
    rmse_cfact = 0
    mae_ate = 0
     
    eff_pred = c.cf.pred - c.pred
    eff_pred[T_test > 0] = -eff_pred[T_test > 0]
     
    ite_pred = YCF_test.hat - YF_test
    ite_pred[T_test > 0] = -ite_pred[T_test > 0]
    rmse_ite = 0
     
    ate_pred = mean(eff_pred)
    bias_ate = 0
     
     
    att_pred = mean(eff_pred[T_test > 0])
    bias_att = 0
     
    atc_pred = mean(eff_pred[T_test < 1])
    bias_atc = 0
     
    policy = eff_pred>0
    treat_overlap = (policy==T_test) * T_test>0
    control_overlap = (policy==T_test) * T_test<1
    
    if(sum(treat_overlap)==0)
    {
    treat_value = 0
    }else
    {
      treat_value = mean(YF_test[treat_overlap])
    }
    if(sum(control_overlap)==0)
    {
    control_value = 0
    }else
    {
    control_value = mean(YF_test[control_overlap])
    }
     
    pit = mean(policy)
    policy_value = (pit * treat_value) + ((1-pit)*control_value)
    pehe = 0
    
    if(dataset=="IHDP")
    {
      YCF_test<-npz1_test['ycf'][,i_exp]
      Y1<-ifelse(T_test>0, YF_test, YCF_test)
      Y0<-ifelse(T_test>0, YCF_test, YF_test)
      YCF<-npz1['ycf'][,i_exp]
      mu1<-npz1_test['mu1'][,i_exp]
      mu0<-npz1_test['mu0'][,i_exp]
      
      y1_pred_ts = ifelse(T_test>0, Y_test.hat, YCF_test.hat)
      y0_pred_ts = ifelse(T_test>0, YCF_test.hat, Y_test.hat)
      
      eff = mu1-mu0
      pehe = sqrt(mean((y1_pred_ts-Y1)+(Y0-y0_pred_ts))^2)
      #pehe = sqrt(mean(((y1_pred_ts-y0_pred_ts)-(Y1-Y0))^2))
    }
    pehe_nn = 0
    policy_curve = 0
    policy_risk = 1 - policy_value
    print(pehe)
    df <- rbind(df, data.frame(model,dataset,rmse_ite,ate_pred,att_pred,bias_att,atc_pred,bias_atc,bias_ate,rmse_fact,policy_curve,policy_value,pehe,pehe_nn,policy_risk))
  }
}
result_df = data.frame(model,dataset,mean(df[["rmse_ite"]]),mean(df[["ate_pred"]]),mean(df[["att_pred"]]),mean(df[["bias_att"]]),mean(df[["atc_pred"]]),mean(df[["bias_atc"]]),mean(df[["bias_ate"]]),mean(df[["rmse_fact"]]),mean(df[["policy_curve"]]),mean(df[["policy_value"]]),mean(df[["pehe"]]),mean(df[["pehe_nn"]]),mean(df[["policy_risk"]]))
write.table(result_df,write_fn,sep = ",",col.names=FALSE,row.names=FALSE,append=TRUE)