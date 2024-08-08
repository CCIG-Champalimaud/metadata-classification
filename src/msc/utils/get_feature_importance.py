import os
import argparse
import numpy as np
import pickle
import pandas as pd
from glob import glob
from tqdm import tqdm

import sys

sys.path.append(".")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", required=True)
    parser.add_argument("--output_path", required=False)
    args = parser.parse_args()

    all_models = {}
    for model in glob(os.path.join(args.input_path, "*pkl")):
        with open(model, "rb") as o:
            model_details = pickle.load(o)
            all_models[os.path.split(model)[-1]] = model_details

    keys = ["ADC", "DCE", "DWI", "Others", "T2W"]
    f_imp_dict = {
        "model": [],
        "exclusion": [],
        "class": [],
        "value": [],
        "feature": [],
        "fold": [],
        "fraction": [],
    }

    for model_k in tqdm(all_models):
        info = model_k.split(".")
        model_name = info[0]
        exclusion_str = info[1]
        fraction = info[2] if len(info) == 4 else 1.0
        cv_models, test_models = (
            all_models[model_k]["cv"],
            all_models[model_k]["test"],
        )
        for i, (f_cv, f_test) in enumerate(zip(cv_models, test_models)):
            feature_keys = np.array(f_cv["feature_names"])
            feature_imps = f_test["feature_importance"]
            if feature_imps is None:
                continue
            if model_name != "catboost":
                rzv = f_cv["model"]["rzv"]
                feature_keys = feature_keys[rzv.variances_ > rzv.threshold]
                feature_imps = feature_imps.mean(1)
            else:
                feature_imps = feature_imps.mean(0)[:, :-1].swapaxes(0, 1)
            for c in range(feature_imps.shape[1]):
                class_f_imp = feature_imps[:, c]
                f_imp_dict["model"].extend(
                    [model_name for _ in range(class_f_imp.shape[0])]
                )
                f_imp_dict["exclusion"].extend(
                    [exclusion_str for _ in range(class_f_imp.shape[0])]
                )
                f_imp_dict["class"].extend(
                    [keys[c] for _ in range(class_f_imp.shape[0])]
                )
                f_imp_dict["fold"].extend(
                    [i for _ in range(class_f_imp.shape[0])]
                )
                f_imp_dict["fraction"].extend(
                    [fraction for _ in range(class_f_imp.shape[0])]
                )
                f_imp_dict["value"].extend(class_f_imp)
                f_imp_dict["feature"].extend(feature_keys)
    pd.DataFrame.from_dict(f_imp_dict).to_csv(args.output_path, index=False)
