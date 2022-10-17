import os
import argparse
import numpy as np
import pickle
import pandas as pd
from glob import glob 
from sklearn.metrics import classification_report,confusion_matrix
from scipy import stats

import sys
sys.path.append('.')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path",required=True)
    parser.add_argument("--output_path",required=True)
    args = parser.parse_args()

    all_models = {}
    for model in glob(os.path.join(args.input_path,"*pkl")):
        with open(model,"rb") as o:
            model_details = pickle.load(o)
            all_models[os.path.split(model)[-1]] = model_details

    keys = ["ADC","DCE","DWI","Others","T2W"]
    metric_dict = {"model":[],"exclusion":[],"metric":[],
                   "value":[],"true":[],"pred":[],"set":[],
                   "split":[],"fold":[]}
    all_test_pred_models = {}
    all_test_y = {}
    for model_k in all_models:
        model_name,exclusion_str,_ = model_k.split(".")
        test_y = None
        all_test_pred = []
        if exclusion_str not in all_test_pred_models:
            all_test_pred_models[exclusion_str] = []
        for X in ["cv","test"]:
            if X in all_models[model_k]:
                for fold_idx,fold in enumerate(all_models[model_k][X]):
                    if X == "test":
                        test_y = fold["y_true"]
                        all_test_pred.append(fold["y_pred"])
                        if model_name in ["xgb","catboost"]:
                            all_test_pred_models[exclusion_str].append(
                                fold["y_pred"])
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
                    # add confusion matrix in long format
                    cm = confusion_matrix(fold["y_true"],fold["y_pred"])
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
                            metric_dict["set"].append("{}_{}".format(k1,k2))
                            metric_dict["split"].append(X)
                            metric_dict["fold"].append(fold_idx)
                            i += 1
                    # add metrics
                    cr = classification_report(fold["y_true"],fold["y_pred"],
                                               output_dict=True)
                    for k in cr:
                        if isinstance(cr[k],dict):
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
        if test_y is not None:
            all_test_pred = np.array(all_test_pred)
            all_test_pred = stats.mode(all_test_pred,axis=0)[0][0]
            cm = confusion_matrix(test_y,all_test_pred)
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
                    metric_dict["set"].append("{}_{}".format(k1,k2))
                    metric_dict["split"].append("test_consensus")
                    metric_dict["fold"].append("all")
                    i += 1
            # add metrics
            cr = classification_report(test_y,all_test_pred,
                                       output_dict=True)
            for k in cr:
                if isinstance(cr[k],dict):
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

    if len(all_test_pred_models) > 0:
        for exclusion_str in all_test_pred_models:
            all_arrays = [np.array(x).flatten() 
                          for x in all_test_pred_models[exclusion_str]]
            all_test_pred = np.array(all_arrays)
            all_test_pred = stats.mode(all_test_pred,axis=0)[0][0]
            cm = confusion_matrix(all_test_y[exclusion_str],all_test_pred)
            i = 0
            cm_r = cm.ravel()
            for k1 in keys:
                for k2 in keys:
                    metric_dict["model"].append("all")
                    metric_dict["exclusion"].append(exclusion_str)
                    metric_dict["metric"].append("cm")
                    metric_dict["value"].append(cm_r[i])
                    metric_dict["true"].append(k1)
                    metric_dict["pred"].append(k2)
                    metric_dict["set"].append("{}_{}".format(k1,k2))
                    metric_dict["split"].append("test_ensemble")
                    metric_dict["fold"].append("all")
                    i += 1
            # add metrics
            cr = classification_report(all_test_y[exclusion_str],all_test_pred,
                                        output_dict=True)
            for k in cr:
                if isinstance(cr[k],dict):
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

    pd.DataFrame.from_dict(metric_dict).to_csv(args.output_path,index=False)