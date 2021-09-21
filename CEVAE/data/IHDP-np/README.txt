These are Python Numpy binaries. Load with numpy.load(...). The fields you need are 'mu1', 'mu0', 't', 'x', 'yf', 'ycf'
mu0 and mu1 are the true potential outcomes (without noise), x are features, t is treatment, yf is observed (factual) outcome, corresponding to t, but with noise. ycf is the counterfactual outcome (with noise). 
