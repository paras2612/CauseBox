install.packages("survival", dependencies=TRUE,repos = "http://cran.us.r-project.org")
install.packages("abind",repos = "http://cran.us.r-project.org")
install.packages("rmatio",repos = "http://cran.us.r-project.org")
install.packages("BART", dependencies=TRUE,repos = "http://cran.us.r-project.org")
install.packages("reticulate", dependencies=TRUE,repos = "http://cran.us.r-project.org")
install.packages('Rcpp',repos = "http://cran.us.r-project.org")
library(Rcpp)
library(BART)
library(rmatio)
library(abind)
library(reticulate)


args <- commandArgs(trailingOnly = TRUE)
fn <- args[1]
test_fn <- args[2]
write_fn <- args[3]

i_exps <- c(1:args[5])

dataset <- args[4]
model = "BART"

trs <- c(0.8)

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
    
    my_bart_f = wbart(XT, YF, x.test=XT_test, nskip=100, ndpost=100)
    my_bart_cf = wbart(XT, YF, x.test=XT_CF_test, nskip=100, ndpost=100)
    
    yf_pred_ts = my_bart_f$yhat.test.mean
    ycf_pred_ts = my_bart_cf$yhat.test.mean
    
    rmse_fact = sqrt(mean((yf_pred_ts - YF_test)**2))
    rmse_cfact = 0
    mae_ate = 0
    
    eff_pred = ycf_pred_ts - yf_pred_ts
    eff_pred[T_test > 0] = -eff_pred[T_test > 0]
    
    ite_pred = ycf_pred_ts - YF_test
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
      
      y1_pred_ts = ifelse(T_test>0, yf_pred_ts, ycf_pred_ts)
      y0_pred_ts = ifelse(T_test>0, ycf_pred_ts, yf_pred_ts)
      
      eff = mu1-mu0
      pehe = sqrt(mean(((y1_pred_ts-y0_pred_ts)-(Y1-Y0))^2))
      mae_ate = abs(mean((yf_pred_ts-ycf_pred_ts))-mean((YF_test-YCF_test)))
      
      rmse_fact = sqrt(mean((yf_pred_ts - YF_test)**2))
      rmse_cfact = sqrt(mean((ycf_pred_ts - YCF_test)**2))
      
      eff_pred = ycf_pred_ts - yf_pred_ts
      eff_pred[T_test > 0] = -eff_pred[T_test > 0]
      
      ite_pred = ycf_pred_ts - YF_test
      ite_pred[T_test > 0] = -ite_pred[T_test > 0]
      rmse_ite = sqrt(mean((ite_pred - eff)**2))
      
      ate_pred = mean(eff_pred)
      bias_ate = ate_pred - mean(eff)
      
      att_pred = mean(eff_pred[T_test > 0])
      bias_att = att_pred - mean(eff[T_test > 0])
      
      atc_pred = mean(eff_pred[T_test < 1])
      bias_atc = atc_pred - mean(eff[T_test < 1])
      
    }
    else{
    pehe_nn = 0
    policy_curve = 0
    policy_risk = 1 - policy_value
    }
    df <- rbind(df, data.frame(model,dataset,rmse_ite,ate_pred,att_pred,bias_att,atc_pred,bias_atc,bias_ate,rmse_fact,policy_curve,policy_value,pehe,pehe_nn,policy_risk))
    
    
  }
}

result_df = data.frame(model,dataset,mean(df[["rmse_ite"]]),mean(df[["ate_pred"]]),mean(df[["att_pred"]]),mean(df[["bias_att"]]),mean(df[["atc_pred"]]),mean(df[["bias_atc"]]),mean(df[["bias_ate"]]),mean(df[["rmse_fact"]]),mean(df[["policy_curve"]]),mean(df[["policy_value"]]),mean(df[["pehe"]]),mean(df[["pehe_nn"]]),mean(df[["policy_risk"]]))
write.table(result_df,write_fn,sep = ",",col.names=FALSE,row.names=FALSE,append=TRUE)
