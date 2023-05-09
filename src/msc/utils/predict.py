import argparse
import json
import dill
import numpy as np
import polars as po
import os
from catboost import FeaturesData
from ..dicom_feature_extraction import (
    extract_features_from_dicom,dicom_header_dict)
from ..constants import text_sep_cols,num_sep_cols,num_cols
from ..sanitization import sanitize_input,replace_cols,rep_dict
from typing import Tuple,List

def read_data(input_paths):
    def read_data_dicom(input_path):
        features = extract_features_from_dicom(input_path)
        features = {k:[features[k]] for k in features}
        features = po.from_dict(features)
        return features

    def read_data_csv_tsv(input_path):
        sep = "\t" if input_path[-3:] == "tsv" else ","
        features = po.read_csv(input_path,sep=sep,ignore_errors=True)
        features.columns = [x.replace(" ","_") for x in features.columns]
        return features

    def read_parquet(input_path):
        features = po.read_parquet(input_path)
        features.columns = [x.replace(" ","_") for x in features.columns]
        return features

    def read_json(input_path):
        json_file = json.load(open(input_path,"r"))
        output_dict = {"patient_id":[],
                       "study_uid":[],
                       "series_uid":[]}
        for patient_id in json_file:
            for study_uid in json_file[patient_id]:
                for series_uid in json_file[patient_id][study_uid]:
                    md_dict = json_file[patient_id][study_uid][series_uid]
                    for metadata_key in md_dict:
                        if all([metadata_key in dicom_header_dict,
                                metadata_key != "patient_id",
                                metadata_key != "study_uid",
                                metadata_key != "series_uid"]):
                            if metadata_key not in output_dict:
                                output_dict[metadata_key] = []
                            values = md_dict[metadata_key]
                            output_dict[metadata_key].extend(values)
                    pid = [patient_id for _ in values]
                    stid = [study_uid for _ in values]
                    seid = [series_uid for _ in values]
                    output_dict["patient_id"].extend(pid)
                    output_dict["study_uid"].extend(stid)
                    output_dict["series_uid"].extend(seid)
        features = po.from_dict(output_dict)
        return features

    all_features = []
    for input_path in input_paths:
        extension = input_path.split(".")[-1]
        if os.path.isdir(input_path) == True:
        # load dicom metadata
            all_features.append(read_data_dicom(input_path))
        elif extension in ["tsv","csv"]:
            all_features.append(read_data_csv_tsv(input_path))
        elif extension == "parquet":
            all_features.append(read_parquet(input_path))
        elif extension == "json":
            all_features.append(read_json(input_path))
        else:
            raise NotImplementedError(
                "Input must be DICOM series dir or csv, tsv or parquet file")
    if len(all_features) > 1:
        all_features = po.concat(all_features,how="vertical")
    else:
        all_features = all_features[0]
    all_features = summarise_columns(all_features)
    return all_features

def summarise_columns(x):
    cols = x.columns
    group_cols = ["study_uid","series_uid"]
    if "patient_id" in x:
        group_cols.append("patient_id")
    col_expressions = [
        (po.col(k)
         .cast(po.Utf8)
         .fill_null("-")
         .str.replace_all(r"^nan$","-")
         .str.replace_all(r"\\"," ")
         .str.replace_all(";"," ")
         .unique()
         .str.concat(" "))
        for k in cols 
        if (k not in group_cols) and (k in dicom_header_dict)]
    if "number_of_images" in x:
        col_expressions.append(
            (po.col("number_of_images")
             .cast(po.Int32)
             .median()))
    elif "number_of_frames" in x:
        col_expressions.append(
            (po.col("number_of_frames")
             .cast(po.Int32)
             .median()
             .alias("number_of_images")))
    elif "number_of_images" not in x:
        col_expressions.append(
            po.col("study_uid").len().alias("number_of_images"))
    output = x.groupby(group_cols).agg(col_expressions)
    return output

def sanitize_input(X,replacements={},strips=[]):
    X = X.cast(po.Utf8)
    for k in replacements:
        X = X.str.replace_all(k,replacements[k])
    for s in strips:
        X = X.str.strip(s)
    X = X.str.to_lowercase()
    return X

def sanitize_data(data):
    col_expr_sep = []
    col_expr_rep = []
    all_sep_cols = text_sep_cols + num_sep_cols
    all_sep_cols = list(set(all_sep_cols))
    for k in all_sep_cols:
        col = sanitize_input(po.col(k),rep_dict,[" "]).alias(k)
        col_expr_sep.append(col)
    for k in replace_cols:
        col = (po.when(po.col(k) == "")
               .then(replace_cols[k])
               .otherwise(po.col(k))
               .alias(k))
        col_expr_rep.append(col)
    data = data.with_columns(col_expr_sep)
    data = data.with_columns(col_expr_rep)
    return data

def get_heuristics(features:po.DataFrame)->Tuple[List[bool],
                                                 List[bool],
                                                 List[bool]]:
    """Gets classification heuristics from feature df.

    Args:
        features (pd.DataFrame): features dataframe containing image_type,
            series_description, diffusion_bvalue, diffusion_bvalue_siemens
            and diffusion_bvalue_ge columns.

    Returns:
        Tuple[List[bool], List[bool], List[bool]]: boolean index vectors
            corresponding to positive classifications for T2W, ADC and DWI
            sequences.
    """
    is_dwi_bool_idx = []
    is_t2w_bool_idx = []
    is_adc_bool_idx = []
    for it,sd,f in zip(features["image_type"].to_list(),
                       features["series_description"].to_list(),
                       features["diffusion_bvalue"].to_list()):
        it = it.lower()
        sd = sd.lower()
        f = [float(x) if x.replace(".","").isnumeric() else 0 
             for x in np.unique(f.split())]
        if len(f) == 0:
            f = 0
        f = np.int32(f).max()
        # check if it can be t2w ("t2" substring in sd AND no 
        # "cor" or "sag" substring)
        if ("t2" in sd) and ("cor" not in sd) and ("sag" not in sd):
            is_t2w_bool_idx.append(True)
        else:
            is_t2w_bool_idx.append(False)
        # check if it can be dwi by seeing whether the maximum 
        # b-value is greater than 0 and that "adc" is not in the
        # series description or image type
        if f > 0 and ("adc" not in sd) and ("adc" not in it):
            is_dwi_bool_idx.append(True)
        else:
            is_dwi_bool_idx.append(False)
        # check if it can be adc ("adc" substring in it)
        if ("adc" in it) or ("adc" in sd.split(" ")):
            is_adc_bool_idx.append(True)
        else:
            is_adc_bool_idx.append(False)
    heuristics_df = po.DataFrame({
        "study_uid":features["study_uid"],
        "series_uid":features["series_uid"],
        "t2w_heuristics":is_t2w_bool_idx,
        "adc_heuristics":is_adc_bool_idx,
        "dwi_heuristics":is_dwi_bool_idx})
    return heuristics_df

def get_consensus_predictions(all_predictions:List[np.ndarray])->List[str]:
    """Calculates consensus predictions (mode) from a list of prediction 
    vectors.

    Args:
        all_predictions (List[np.ndarray]): list of prediction vectors.

    Returns:
        List[str]: consensus predictions.
    """
    consensus_pred = po.DataFrame(
        np.concatenate(all_predictions,axis=1))
    consensus_pred = consensus_pred.to_pandas().mode(1).iloc[:,0]
    consensus_pred = [x.upper() for x in consensus_pred]
    return consensus_pred

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Predicts the type of sequence.")
    
    parser.add_argument("--input_paths",required=True,nargs="+",
                        help="Path to DICOM directory or CSV/TSV containing data")
    parser.add_argument("--model_paths",required=False,nargs="+",
                        help="Path to model")

    args = parser.parse_args()
    
    # if it's a directory, assume it's a directory containing .dcm files
    features = read_data(args.input_paths)
    features = sanitize_data(features)
    features = features.sort(by=["study_uid","series_uid"])

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
    
    # predict
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
    if "patient_id" in features:
        patient_id = features["patient_id"]
    else:
        patient_id = features["study_uid"]
    predictions_df = (po.from_dict({
        "patient_id":patient_id,
        "study_uid":features["study_uid"],
        "series_uid":features["series_uid"],
        "prediction":consensus_pred}).join(heuristics_df,
                                           on=["study_uid","series_uid"])
        .with_columns(po.col("prediction").alias("prediction_heuristics"))
        .with_columns(po.when(po.col("t2w_heuristics") == True)
                      .then("T2")
                      .when(po.col("dwi_heuristics") == True)
                      .then("DWI")
                      .when(po.col("adc_heuristics") == True)
                      .then("ADC")
                      .otherwise(po.col("prediction_heuristics"))
                      .alias("prediction_heuristics"))
        .select(["patient_id","study_uid","series_uid",
                 "prediction","prediction_heuristics"]))
    
    # print predctions
    print(predictions_df.to_pandas().to_csv(index=False))
