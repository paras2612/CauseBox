import sys
import os

import pickle
import pandas as pd
from SITE.logger import Logger as Log
Log.VERBOSE = True

import SITE.evaluation as evaluation
from SITE.plotting import *

def sort_by_config(results, configs, key):
    vals = np.array([cfg[key] for cfg in configs])
    I_vals = np.argsort(vals)

    for k in results['train'].keys():
        results['train'][k] = results['train'][k][I_vals,]
        results['valid'][k] = results['valid'][k][I_vals,]

        if k in results['test']:
            results['test'][k] = results['test'][k][I_vals,]

    configs_sorted = []
    for i in I_vals:
        configs_sorted.append(configs[i])

    return results, configs_sorted

def load_config(config_file):
    with open(config_file, 'r') as f:
        cfg = [l.split('=') for l in f.read().split('\n') if '=' in l]
        cfg = dict([(kv[0], kv[1]) for kv in cfg])
    return cfg

def evaluate(config_file, overwrite=False, filters=None):

    if not os.path.isfile(config_file):
        raise Exception('Could not find config file at path: %s' % config_file)

    cfg = load_config(config_file)
    #print(cfg['outdir'])
    output_dir = cfg['outdir'].strip()
    if not os.path.isdir(output_dir):
        raise Exception('Could not find output at path: %s' % output_dir)

    data_train = cfg['datadir'].strip()+'/'+cfg['dataform'].strip()
    data_test = cfg['datadir'].strip()+'/'+cfg['data_test'].strip()
    binary = False
    if cfg['loss'] == 'log':
        binary = True

    # Evaluate results_try1
    eval_path = '%s/evaluation.npz' % output_dir
    if overwrite or (not os.path.isfile(eval_path)):
        eval_results, configs = evaluation.evaluate(output_dir,
                                data_path_train=data_train,
                                data_path_test=data_test,
                                binary=binary)
        # Save evaluation
        pickle.dump((eval_results, configs), open(eval_path, "wb"))
    else:
        #if Log.VERBOSE:
            #print('Loading evaluation results_try1 from %s...' % eval_path)
        # Load evaluation
        eval_results, configs = pickle.load(open(eval_path, "rb"))
    # print eval_results
    # print configs
    #data_train_find = load_data(data_train)
    res_dict = {}
    res_dict["model"] = "SITE"
    res_dict['dataset'] = cfg['dataset']
    try:
        res_dict['rmse_ite'] = float(np.mean(np.abs(eval_results['test'].get("rmse_ite"))))
    except:
        res_dict['rmse_ite'] = float(0)
    try:
        res_dict['ate_pred'] = float(np.mean(np.abs(eval_results['test'].get("ate_pred"))))
    except:
        res_dict['ate_pred'] = float(0)
    try:
        res_dict['att_pred'] = float(np.mean(np.abs(eval_results['test'].get("att_pred"))))
    except:
        res_dict['att_pred'] = float(0)
    try:
        res_dict['bias_att'] = float(np.mean(np.abs(eval_results['test'].get("bias_att"))))
    except:
        res_dict['bias_att'] = float(0)
    try:
        res_dict['atc_pred'] = float(np.mean(np.abs(eval_results['test'].get("atc_pred"))))
    except:
        res_dict['atc_pred'] = float(0)
    try:
        res_dict['bias_ate'] = float(np.mean(np.abs(eval_results['test'].get("bias_ate"))))
    except:
        res_dict['bias_ate'] = float(0)
    try:
        res_dict['bias_atc'] = float(np.mean(np.abs(eval_results['test'].get("bias_atc"))))
    except:
        res_dict['bias_atc'] = float(0)
    try:
        res_dict['rmse_fact'] = float(np.mean(np.abs(eval_results['test'].get("rmse_fact"))))
    except:
        res_dict['rmse_fact'] = float(0)
    try:
        res_dict['policy_value'] = float(np.mean(np.abs(eval_results['test'].get("policy_value"))))
    except:
        res_dict['policy_value'] = float(0)
    try:
        res_dict['policy_curve'] = float(np.mean(np.abs(eval_results['test'].get("policy_curve"))))
    except:
        res_dict['policy_curve'] = float(0)
    try:
        res_dict['pehe'] = float(np.mean(np.abs(eval_results['test'].get("pehe"))))
    except:
        res_dict['pehe'] = float(0)
    try:
        res_dict['pehe_nn'] = float(np.mean(np.abs(eval_results['test'].get("pehe_nn"))))
    except:
        res_dict['pehe_nn'] = float(0)
    try:
        res_dict['policy_risk'] = float(np.mean(np.abs(eval_results['test'].get("policy_risk"))))
    except:
        res_dict['policy_risk'] = float(0)
    res = pd.DataFrame.from_dict([res_dict])
    res.columns = ["MODEL", "DATASET", "RMSE_ITE", "ATE_PRED", "ATT_PRED", "BIAS_ATT", "ATC_PRED", "BIAS_ATC",
                   "BIAS_ATE", "RMSE_FACT", "POLICY_CURVE", "POLICY_VALUE", "PEHE", "PEHE_NN", "POLICY_RISK"]
    fname = cfg["datadir"].strip() + cfg["data_test"].strip()
    if fname[-3:] == 'npz':
        data_in = np.load(fname)
    if (cfg["dataset"].lower().strip() == "ihdp"):
        ate = np.mean(data_in["mu1"] - data_in["mu0"])
        error_ate = np.abs(res["ATE_PRED"].values[0] - ate)
        print("PEHE is ", round(res["PEHE"].values[0], 2), "and error in ATE is ", round(error_ate, 2))
    else:
        att = np.mean(data_in["ate"])
        error_att = np.abs(res["ATT_PRED"].values[0] - att)
        print("Policy Risk is ", round(res["POLICY_RISK"].values[0], 2), "and error in ATT is ", round(error_att, 2))
    filename = os.getcwd() + "\Results.csv"
    res.to_csv(filename, mode="a", header=False, index=False)



    '''if binary:
        if data_train_find['HAVE_TRUTH']:
            plot_evaluation_cont(eval_results, configs, output_dir, data_train, data_test, filters)
        else:

            plot_evaluation_bin(eval_results, configs, output_dir, data_train, data_test, filters)
    else:
        plot_evaluation_cont(eval_results, configs, output_dir, data_train, data_test, filters)
        '''


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('Usage: python evaluate.py <config_file> <overwrite (default 0)> <filters (optional)>')
    else:
        config_file = sys.argv[1]

        overwrite = False
        if len(sys.argv)>2 and sys.argv[2] == '1':
            overwrite = True

        filters = None
        if len(sys.argv)>3:
            filters = eval(sys.argv[3])

        evaluate(config_file, overwrite, filters=filters)
