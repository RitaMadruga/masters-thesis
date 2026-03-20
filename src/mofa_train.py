from mofapy2.run.entry_point import entry_point
import pandas as pd

mrna_data_lg2 = pd.read_csv("data/transformed_data/mrna_data_lg2.csv", index_col=0)
mrna_data_vsn = pd.read_csv("data/transformed_data/mrna_data_vsn.csv", index_col=0)
mirna_data_lg2 = pd.read_csv("data/transformed_data/mirna_data_lg2.csv", index_col=0)
mirna_data_vsn = pd.read_csv("data/transformed_data/mirna_data_vsn.csv", index_col=0)
meth_data = pd.read_csv("data/transformed_data/meth_data_m_values.csv", index_col=0)

mrna_data_lg2_fs = pd.read_csv("data/feature_selection/selected_features_mrna_data_lg2.csv", index_col=0)
mrna_data_vsn_fs = pd.read_csv("data/feature_selection/selected_features_mrna_data_vsn.csv", index_col=0)
mirna_data_lg2_fs = pd.read_csv("data/feature_selection/selected_features_mirna_data_lg2.csv", index_col=0)
mirna_data_vsn_fs = pd.read_csv("data/feature_selection/selected_features_mirna_data_vsn.csv", index_col=0)
meth_data_fs = pd.read_csv("data/feature_selection/selected_features_meth_data.csv", index_col=0)

runs_config = {"FeatureSelected_LG": {"datasets": [mrna_data_lg2_fs, mirna_data_lg2_fs, meth_data_fs], "save_path": "data/latent/mofa_trained_lg2_fs.hdf5"},
                "FeatureSelected_VSN": {"datasets": [mrna_data_vsn_fs, mirna_data_vsn_fs, meth_data_fs], "save_path": "data/latent/mofa_trained_vsn_fs.hdf5"},
                "AllFeatures_LG": {"datasets": [mrna_data_lg2, mirna_data_lg2, meth_data], "save_path": "data/latent/mofa_trained_lg2.hdf5"},
                "AllFeatures_VSN": {"datasets": [mrna_data_vsn, mirna_data_vsn, meth_data], "save_path": "data/latent/mofa_trained_vsn.hdf5"}}

for run_name, config in runs_config.items():

    mrna, mirna, meth = config["datasets"]
    save_path = config["save_path"]

    common_samples = mrna.index.intersection(mirna.index).intersection(meth.index)

    if common_samples.empty:
        raise ValueError(f"No common samples found for run {run_name}")

    mrna = mrna.loc[common_samples]
    mirna = mirna.loc[common_samples]
    meth = meth.loc[common_samples]

    ent = entry_point()
    data_matrix = [[mrna.values], [mirna.values], [meth.values]]

    views = ["mRNA", "miRNA", "Methylation"]
    samples = [mrna.index.tolist()]
    features = [mrna.columns.tolist(), mirna.columns.tolist(), meth.columns.tolist()]

    ent.set_data_options(scale_groups=False, scale_views=True)

    ent.set_data_matrix(data_matrix, likelihoods=["gaussian", "gaussian", "gaussian"], views_names=views, groups_names = ['PAAD'],samples_names=samples, features_names=features)

    ent.set_model_options(factors = 10, spikeslab_weights = True, ard_weights=True, ard_factors=False)
    ent.set_train_options(iter=1000, convergence_mode="medium", freqELBO=10, gpu_mode=True, seed = 123)

    ent.build()
    ent.run()
    ent.save(save_path)