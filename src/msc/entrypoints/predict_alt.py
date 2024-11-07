import argparse
import dill
import numpy as np
import polars as po
import os
from scipy.stats import mode
from catboost import FeaturesData
from ..dicom_feature_extraction import extract_features_from_dicom,pcai_mapping
from ..constants import text_sep_cols,num_sep_cols,num_cols

from .predict import (
    summarise_columns,sanitize_data,
    get_heuristics,get_consensus_predictions)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Predicts the type of sequence.")
    
    parser.add_argument("--input_path",required=True,
                        help="Path to DICOM directory or CSV/TSV containing data")
    parser.add_argument("--model_paths",required=False,nargs="+",
                        help="Path to model")

    args = parser.parse_args()
    
    # if it's a directory, assume it's a directory containing .dcm files
    if os.path.isdir(args.input_path) == True:
        # load dicom headers
        features = extract_features_from_dicom(args.input_path)
        features = {k:[features[k]] for k in features}
        features = po.from_dict(features)
    elif args.input_path.split(".")[-1] in ["tsv","csv"]:
        sep = "\t" if args.input_path[-3:] == "tsv" else ","
        features = po.read_csv(args.input_path,sep=sep,ignore_errors=True)
        features.columns = [pcai_mapping[k] if k in pcai_mapping else k 
                            for k in features.columns]
        features = features.with_columns(po.lit("-").alias("diffusion_directionality"))
        features.columns = [x.replace(" ","_") for x in features.columns]
        features = summarise_columns(features)
    elif args.input_path.split(".")[-1] == "parquet":
        features = po.read_parquet(args.input_path)
        features.columns = [x.replace(" ","_") for x in features.columns]
        features = summarise_columns(features)
    features = sanitize_data(features)
    
    # setup models
    all_predictions = []
    match = ["ADC","DCE","DWI","Others","T2"]
    all_predictions_fold = []
    all_series_uid = features["series_uid"]
    all_study_uid = features["study_uid"]
    if "patient_id" in features:
        all_patient_id = features["patient_id"]
    else:
        all_patient_id = features["study_uid"]
    for model_path in args.model_paths:
        model = dill.load(open(model_path,"rb"))
        for fold in model["cv"]:
            if fold["count_vec"] is not None:
                is_catboost = False
                count_vec = fold["count_vec"]
                transformed_features = count_vec.transform(features)
                prediction = fold["model"].predict(transformed_features)
                all_predictions_fold.append(match[int(prediction)].upper())
            else:
                is_catboost = True
                fc = text_sep_cols + num_sep_cols
                text_arr = []
                num_arr = []
                for f in fold["feature_names"]:
                    if f in fc:
                        text_arr.append(features[f].to_numpy())
                    elif f in num_cols:
                        num_arr.append(features[f].to_numpy())
                text_arr = np.array(text_arr).T
                num_arr = np.array(num_arr).T
                dat = FeaturesData(num_feature_data=num_arr.astype(np.float32),
                                   cat_feature_data=text_arr)
                prediction = fold["model"].predict(dat)
                all_predictions_fold.append(prediction.astype(str))

    # aggregate prediction consensus
    consensus_pred = get_consensus_predictions(all_predictions_fold)
    
    # calculate heuristics
    heuristics_df = get_heuristics(features)
    
    # merge predictions with heuristics
    predictions_df = (po.from_dict({
        "patient_id":features["series_uid"],
        "study_uid":features["study_uid"],
        "series_uid":features["series_uid"],
        "prediction":consensus_pred}).join(heuristics_df,
                                           on=["study_uid","series_uid"])
        .with_columns(po.col("prediction").alias("prediction_heuristics"))
        .with_columns(
            [po.when(po.col("t2w_heuristics") == True)
             .then("T2")
             .otherwise(po.col("prediction_heuristics"))
             .alias("prediction_heuristics"),
             po.when(po.col("dwi_heuristics") == True)
             .then("DWI")
             .otherwise(po.col("prediction_heuristics"))
             .alias("prediction_heuristics"),
             po.when(po.col("adc_heuristics") == True)
             .then("ADC")
             .otherwise(po.col("prediction_heuristics"))
             .alias("prediction_heuristics")]
        )
        .select(["patient_id","study_uid","series_uid",
                 "prediction","prediction_heuristics"]))
    
    # print predctions
    print(predictions_df.to_pandas().to_csv(header=None,index=False))
