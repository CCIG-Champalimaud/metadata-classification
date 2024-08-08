import os

n_folds = 5
n_jobs = 4
seed = 42
output_paths = []

data_full = "data/data_full.tsv"
data_train = "data/data_train.csv"
data_test = "data/data_test.csv"
output_metrics = "data_output/metrics.csv"
output_metrics_logo = "data_output/metrics_logo.csv"
output_features = "data_output/feature_importances.csv"
models_dir = "/storage/models/metadata/models"
models_dir_logo = "/storage/models/metadata/models_logo"
group_folds = "data/dataset_split.json"

os.makedirs("models",exist_ok=True)
os.makedirs(os.path.dirname(output_metrics),exist_ok=True)

exclusions_list = [
    [],
    ["series_description"],
    ["percent_phase_field_of_view","sar"],
    ["percent_phase_field_of_view","sar","series_description"]
    ]
model_list = [
    "xgb",
    "catboost",
    ]

exclusion_dict = {}
fractions = [1, 2, 5, 10, 25, 50, 70, 100]
for fraction in fractions:
    for model_name in model_list:
        for exclusion in exclusions_list:
            if len(exclusion) == 0:
                exclusion_str = "standard"
                exclusion_flag = ""
            else:
                exclusion_str = ":".join(exclusion)
                exclusion_flag = "--exclude_cols {}".format(" ".join(exclusion))
            output_path = "{}/{}.{}.{}.pkl".format(
                models_dir,model_name,exclusion_str,fraction)
            exclusion_dict[exclusion_str] = exclusion_flag
            output_paths.append(output_path)


for model_name in model_list:
    for exclusion in exclusions_list:
        if len(exclusion) == 0:
            exclusion_str = "standard"
            exclusion_flag = ""
        else:
            exclusion_str = ":".join(exclusion)
            exclusion_flag = "--exclude_cols {}".format(" ".join(exclusion))
        output_path = "{}/{}.{}.pkl".format(
            models_dir_logo,model_name,exclusion_str)
        exclusion_dict[exclusion_str] = exclusion_flag
        output_paths.append(output_path)

rule all:
    input: output_paths,output_metrics,output_features, output_metrics_logo

rule split_data:
    input:
        data_full
    output:
        data_train=data_train,
        data_test=data_test
    params:
        sep='"\t"',
        id_col="study_uid"
    shell:
        """
        python3 src/msc/utils/split_csv.py \
            --input_path {input} \
            --output_paths {output.data_train} {output.data_test} \
            --sep {params.sep} \
            --random_seed {seed} \
            --id_col {params.id_col}
        """

rule train_models:
    input: 
        data_train=data_train,
        data_test=data_test
    output:
        models_dir + "/{model_name}.{exclusion_str}.{fraction}.pkl"
    params:
        exclusion_flag=lambda wc: exclusion_dict[wc.exclusion_str],
        fraction = lambda wc: int(wc.fraction) / 100,
        save_model_flag = lambda wc: "--save_model" if wc.fraction == '100' else "",
        feature_importance_flag = lambda wc: "" if wc.fraction == '100' else "--skip_feature_importance",
    shell:
        """
        python3 -m src.msc \
            --input_path {input.data_train} \
            --output_path {output} \
            --n_folds {n_folds} \
            --test_set_path {input.data_test} \
            --model_name {wildcards.model_name} \
            --random_seed {seed} \
            --subset {params.fraction} \
            --n_jobs {n_jobs} {params.exclusion_flag} {params.save_model_flag} {params.feature_importance_flag}
        """

rule train_models_logo:
    input: 
        data_train=data_train,
        data_test=data_test
    output:
        models_dir_logo + "/{model_name}.{exclusion_str}.pkl"
    params:
        exclusion_flag=lambda wc: exclusion_dict[wc.exclusion_str],
        save_model_flag = "",
        feature_importance_flag = "--skip_feature_importance"
    shell:
        """
        python3 -m src.msc \
            --input_path {input.data_train} \
            --output_path {output} \
            --folds {group_folds} \
            --test_set_path {input.data_test} \
            --model_name {wildcards.model_name} \
            --random_seed {seed} \
            --n_jobs {n_jobs} {params.exclusion_flag} {params.save_model_flag} {params.feature_importance_flag}
        """

rule collect_all_metrics:
    input: 
        output_paths
    output:
        output_metrics
    shell:
        """
        python3 src/msc/utils/collect_metrics.py \
            --input_path {models_dir} \
            --output_path {output}
        """

rule collect_all_metrics_logo:
    input: 
        output_paths
    output:
        output_metrics_logo
    shell:
        """
        python3 src/msc/utils/collect_metrics.py \
            --input_path {models_dir_logo} \
            --output_path {output}
        """


rule collect_all_feature_importances:
    input:
        output_paths
    output:
        output_features
    shell:
        """
        python3 src/msc/utils/get_feature_importance.py \
            --input_path {models_dir} \
            --output_path {output}
        """
