import os
import argparse
import numpy as np
import pickle
import pandas as pd
from glob import glob
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm

import sys

sys.path.append(".")


def mode(array: np.ndarray):
    u, c = np.unique(array, return_counts=True)
    idx = np.argmax(c)
    return u[idx]


def mode_array(array: np.ndarray):
    output = np.zeros(array.shape[1], dtype=object)
    for i in range(array.shape[1]):
        output[i] = mode(array[:, i])
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", required=True)
    parser.add_argument("--output_path", required=True)
    args = parser.parse_args()

    all_models = {}
    for model in glob(os.path.join(args.input_path, "*pkl")):
        with open(model, "rb") as o:
            model_details = pickle.load(o)
            all_models[os.path.split(model)[-1]] = model_details

    keys = ["ADC", "DCE", "DWI", "Others", "T2W"]
    keys_real = ["adc", "dce", "dwi", "others", "t2"]
    metric_dict = {
        "model": [],
        "exclusion": [],
        "metric": [],
        "value": [],
        "true": [],
        "pred": [],
        "set": [],
        "split": [],
        "fold": [],
        "fraction": [],
    }
    all_test_pred_models = {}
    all_test_y = {}
    for model_k in tqdm(all_models):
        info = model_k.split(".")
        model_name = info[0]
        exclusion_str = info[1]
        fraction = info[2] if len(info) == 4 else 1.0
        test_y = None
        all_test_pred = []
        if exclusion_str not in all_test_pred_models:
            all_test_pred_models[exclusion_str] = {}
        if fraction not in all_test_pred_models[exclusion_str]:
            all_test_pred_models[exclusion_str][fraction] = []
        for X in ["cv", "test"]:
            if X in all_models[model_k]:
                for fold in all_models[model_k][X]:
                    fold_idx = fold["fold"]
                    if X == "test":
                        test_y = fold["y_true"]
                        all_test_pred.append(fold["y_pred"])
                        if model_name in ["xgb", "catboost"]:
                            all_test_pred_models[exclusion_str][
                                fraction
                            ].append(fold["y_pred"])
                            all_test_y[exclusion_str] = fold["y_true"]
                    # add auc
                    metric_dict["model"].append(model_name)
                    metric_dict["exclusion"].append(exclusion_str)
                    metric_dict["metric"].append("auc")
                    metric_dict["value"].append(fold["auc"])
                    metric_dict["true"].append(np.nan)
                    metric_dict["pred"].append(np.nan)
                    metric_dict["set"].append("full")
                    metric_dict["split"].append(X)
                    metric_dict["fold"].append(fold_idx)
                    metric_dict["fraction"].append(fraction)
                    # add confusion matrix in long format
                    cm = confusion_matrix(
                        fold["y_true"], fold["y_pred"], labels=keys_real
                    )
                    i = 0
                    cm_r = cm.ravel()
                    for k1 in keys:
                        for k2 in keys:
                            metric_dict["model"].append(model_name)
                            metric_dict["exclusion"].append(exclusion_str)
                            metric_dict["metric"].append("cm")
                            metric_dict["value"].append(cm_r[i])
                            metric_dict["true"].append(k1)
                            metric_dict["pred"].append(k2)
                            metric_dict["set"].append("{}_{}".format(k1, k2))
                            metric_dict["split"].append(X)
                            metric_dict["fold"].append(fold_idx)
                            metric_dict["fraction"].append(fraction)
                            i += 1
                    # add metrics
                    cr = classification_report(
                        fold["y_true"],
                        fold["y_pred"],
                        output_dict=True,
                        zero_division=np.nan,
                    )
                    for k in cr:
                        if isinstance(cr[k], dict):
                            for kk in cr[k]:
                                metric_dict["model"].append(model_name)
                                metric_dict["exclusion"].append(exclusion_str)
                                metric_dict["metric"].append(kk)
                                metric_dict["value"].append(cr[k][kk])
                                metric_dict["true"].append(np.nan)
                                metric_dict["pred"].append(np.nan)
                                metric_dict["set"].append(k)
                                metric_dict["split"].append(X)
                                metric_dict["fold"].append(fold_idx)
                                metric_dict["fraction"].append(fraction)
        if test_y is not None:
            all_test_pred = np.array(all_test_pred)
            all_test_pred = mode_array(all_test_pred)
            if len(all_test_pred.shape) > 1:
                all_test_pred = all_test_pred[:, 0]
            cm = confusion_matrix(test_y, all_test_pred, labels=keys_real)
            i = 0
            cm_r = cm.ravel()
            for k1 in keys:
                for k2 in keys:
                    metric_dict["model"].append(model_name)
                    metric_dict["exclusion"].append(exclusion_str)
                    metric_dict["metric"].append("cm")
                    metric_dict["value"].append(cm_r[i])
                    metric_dict["true"].append(k1)
                    metric_dict["pred"].append(k2)
                    metric_dict["set"].append("{}_{}".format(k1, k2))
                    metric_dict["split"].append("test_consensus")
                    metric_dict["fold"].append("all")
                    metric_dict["fraction"].append(fraction)
                    i += 1
            # add metrics
            cr = classification_report(
                test_y, all_test_pred, output_dict=True, zero_division=np.nan
            )
            for k in cr:
                if isinstance(cr[k], dict):
                    for kk in cr[k]:
                        metric_dict["model"].append(model_name)
                        metric_dict["exclusion"].append(exclusion_str)
                        metric_dict["metric"].append(kk)
                        metric_dict["value"].append(cr[k][kk])
                        metric_dict["true"].append(np.nan)
                        metric_dict["pred"].append(np.nan)
                        metric_dict["set"].append(k)
                        metric_dict["split"].append("test_consensus")
                        metric_dict["fold"].append("all")
                        metric_dict["fraction"].append(fraction)

    if len(all_test_pred_models) > 0:
        for exclusion_str in all_test_pred_models:
            for fraction in all_test_pred_models[exclusion_str]:
                all_arrays = [
                    np.array(x).flatten()
                    for x in all_test_pred_models[exclusion_str][fraction]
                ]
                all_test_pred = np.array(all_arrays).astype(str)
                all_test_pred = mode_array(all_test_pred)
                if len(all_test_pred.shape) > 1:
                    all_test_pred = all_test_pred[:, 0]
                cm = confusion_matrix(
                    all_test_y[exclusion_str], all_test_pred, labels=keys_real
                )
                i = 0
                print(cm)
                cm_r = cm.ravel()
                for k1 in keys:
                    for k2 in keys:
                        metric_dict["model"].append("all")
                        metric_dict["exclusion"].append(exclusion_str)
                        metric_dict["metric"].append("cm")
                        metric_dict["value"].append(cm_r[i])
                        metric_dict["true"].append(k1)
                        metric_dict["pred"].append(k2)
                        metric_dict["set"].append("{}_{}".format(k1, k2))
                        metric_dict["split"].append("test_ensemble")
                        metric_dict["fold"].append("all")
                        metric_dict["fraction"].append(fraction)
                        i += 1
                # add metrics
                cr = classification_report(
                    all_test_y[exclusion_str],
                    all_test_pred,
                    output_dict=True,
                    zero_division=np.nan,
                )
                for k in cr:
                    if isinstance(cr[k], dict):
                        for kk in cr[k]:
                            metric_dict["model"].append("all")
                            metric_dict["exclusion"].append(exclusion_str)
                            metric_dict["metric"].append(kk)
                            metric_dict["value"].append(cr[k][kk])
                            metric_dict["true"].append(np.nan)
                            metric_dict["pred"].append(np.nan)
                            metric_dict["set"].append(k)
                            metric_dict["split"].append("test_ensemble")
                            metric_dict["fold"].append("all")
                            metric_dict["fraction"].append(fraction)

    pd.DataFrame.from_dict(metric_dict).to_csv(args.output_path, index=False)
