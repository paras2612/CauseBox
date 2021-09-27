class CFRNet:
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
                "imb_fun",
                "output_csv",
                "dataset"]
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
                "repetitions",
                "use_p_correction",
                "optimizer",
                "output_delay",
                "pred_output_delay",
                "debug",
                "save_rep",
                "val_part",
                "split_output",
                "seed",
                "reweight_sample"]


class CEVAE:
    required = ["Num_data",
                "Feature_dim",
                "Latent_dim",
                "Hidden_dim",
                "Num_layers",
                "Epochs",
                "Batch_size",
                "Lr",
                "Lrate_decay",
                "Weight_decay",
                "Seed",
                "Train_path",
                "Test_path",
                "weight-decay",
                "learning-rate",
                "hidden-dim",
                "batch-size",
                "z-dim",
                "hidden-layers",
                "dataset"]

    optional = []

class BART:
    required = ["Rscript_Path",
                "Train_data_path",
                "Test_data_path",
                "Out_data_path",
                "dataset",
                "Experiments"]
    optional = []


class CausalForests:
    required = ["Rscript_Path",
                "Train_data_path",
                "Test_data_path",
                "Out_data_path",
                "dataset",
                "Experiments"]
    optional = []


class PerfectMatch:
    required = ["Seed",
                "Output_directory",
                "Load_existing",
                "N_jobs",
                "Learning_rate",
                "L2_weight",
                "Num_epochs",
                "Batch_size",
                "Early_stopping_patience",
                "Num_units",
                "Num_layers",
                "Dropout",
                "Imbalance_loss_weight",
                "Fraction_of_data_set",
                "Validation_set_fraction",
                "Test_set_fraction",
                "Num_hyperopt_runs",
                "Hyperopt_offset",
                "Tcga_num_features",
                "Experiment_index",
                "Num_treatments",
                "Num_randomised_neighbors",
                "Strength_of_assignment_bias",
                "Propensity_batch_probability",
                "Ganite_weight_alpha",
                "Ganite_weight_beta",
                "Benchmark",
                "Method",
                "With_rnaseq",
                "Do_not_use_tarnet",
                "Do_hyperopt",
                "Do_evaluate",
                "Hyperopt_against_eval_set",
                "Copy_to_local",
                "Do_hyperopt_on_lsf",
                "Do_merge_lsf",
                "With_tensorboard",
                "With_propensity_dropout",
                "With_propensity_batch",
                "Early_stopping_on_pehe",
                "With_pehe_loss",
                "Match_on_covariates",
                "dataset",
                "Save_predictions"]
    optional = []

class DRNet:
    required = ["Loss",
                "P_alpha",
                "P_lambda",
                "P_Beta",
                "Lrate",
                "Batch_size",
                "Experiments",
                "Iterations",
                "Outdir",
                "Datadir",
                "Dataform",
                "Data_test",
                "output_csv",
                "dataset",
                "imb_fun"]

    optional = ["N_in",
                "N_out",
                "Rep_weight_decay",
                "Dropout_in",
                "Dropout_out",
                "Nonlin",
                "Decay",
                "Dim_in",
                "Dim_out",
                "Batch_norm",
                "Normalization",
                "Rbf_Sigma",
                "Weight_init",
                "Lrate_decay",
                "Wass_iterations",
                "Wass_lambda",
                "Wass_bpt",
                "Varsel",
                "Sparse",
                "Seed",
                "Repetitions",
                "Use_P_correction",
                "Optimizer",
                "Output_delay",
                "Pred_output_delay",
                "Debug",
                "Save_rep",
                "Val_part",
                "Split_output",
                "Reweight_sample"]


class SITE:
    required = ["Loss",
                "P_pddm",
                "P_mid_point_mini",
                "Propensity_dir",
                "P_lambda",
                "Lrate",
                "Batch_size",
                "Experiments",
                "Iterations",
                "Outdir",
                "Datadir",
                "Dataform",
                "Data_test",
                "dataset",
                "output_csv"]

    optional = ["N_in",
                "N_out",
                "Rep_weight_decay",
                "Dropout_in",
                "Dropout_out",
                "Nonlin",
                "Decay",
                "Dim_in",
                "Dim_out",
                "Batch_norm",
                "Normalization",
                "Rbf_Sigma",
                "Weight_init",
                "Lrate_decay",
                "Wass_iterations",
                "Wass_lambda",
                "Wass_bpt",
                "Varsel",
                "Sparse",
                "Seed",
                "Repetitions",
                "Use_P_correction",
                "Optimizer",
                "Output_delay",
                "Pred_output_delay",
                "Debug",
                "Save_rep",
                "Val_part",
                "Split_output",
                "Reweight_sample",
                "Dim_c",
                "Dim_s",
                "Equal_sample",
                "imb_fun"]


if __name__ == "__main__":
    print(CFRNet().opteional)
