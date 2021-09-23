class Model:
    required = ["loss",
                "p_alpha",
                "p_lambda",
                "lrate",
                "batch_size",
                "experiments",
                "iterations",
                "outdir",
                "datadir",
                "dataform",
                "data_test",
                "imb_fun"]
    optional = ["n_in",
                "n_out",
                "rep_weight_decay",
                "dropout_in",
                "dropout_out",
                "nonlin",
                "decay",
                "dim_in",
                "dim_out",
                "batch_norm",
                "normalization",
                "rbf_sigma",
                "weight_init",
                "lrate_decay",
                "wass_iterations",
                "wass_lambda",
                "wass_bpt",
                "varsel",
                "sparse",
                "seed",
                "repetitions",
                "use_p_correction",
                "optimizer",
                "output_delay",
                "pred_output_delay",
                "debug",
                "save_rep",
                "val_part",
                "split_output",
                "reweight_sample"]
filename = "CFRNET_cfg.txt"
params= Model.required + Model.optional
result = dict()
keys = list()
with open(filename, "r") as f:
    text = "".join(f.readlines())
    delimParamsList = [option.split() for option in text.split("\n")]
    delimParamsList = [ele for ele in delimParamsList if ele != []]

    for param in delimParamsList:
        key = param[0].lower()
        val = param[1]
        result[key] = val
        keys.append(key)
orgSet = set(params)
cmpSet = set(keys)

print(orgSet)
print(cmpSet)
print(orgSet.difference(cmpSet))

