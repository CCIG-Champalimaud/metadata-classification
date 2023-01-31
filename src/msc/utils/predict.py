import argparse
import dill
import numpy as np
import polars as po
import os
from scipy.stats import mode
from catboost import FeaturesData
from ..dicom_feature_extraction import extract_features_from_dicom
from ..constants import text_sep_cols,num_sep_cols,num_cols
from ..sanitization import sanitize_input,replace_cols,rep_dict
from typing import Tuple,List

def process_column(x):
    output = {}
    for k in x.columns:
        O = []
        for y in x[k]:
            y = str(y).replace(";"," ")
            if y == "nan":
                y = "-"
            if y not in O:
                O.append(y)
        output[k] = [" ".join(O)]
    output["number_of_images"] = x.shape[0]
    return po.from_dict(output)

def sanitize_data(data):
    col_rep = []
    col_rep.extend([
        po.Series(name=k,
                  values=sanitize_input(data[k],rep_dict,[" "]))
        for k in text_sep_cols])
    col_rep.extend([
        po.Series(name=k,
                  values=sanitize_input(data[k],rep_dict,[" "]))
        for k in num_sep_cols])
    col_rep.extend([
        po.when(data[k] == "").then(replace_cols[k]).otherwise(data[k])
        for k in replace_cols])
    data = data.with_columns(col_rep)
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
        if "adc" in it:
            is_adc_bool_idx.append(True)
        else:
            is_adc_bool_idx.append(False)
    return is_t2w_bool_idx,is_adc_bool_idx,is_dwi_bool_idx

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
        number_of_images = features.shape[0]
        features.columns = [x.replace(" ","_") for x in features.columns]
        features = features.groupby(["study_uid","series_uid"]).apply(
            process_column)
    features = sanitize_data(features)
    
    # calculate heuristics
    heuristics = get_heuristics(features)
    is_t2w_bool_idx = heuristics[0]
    is_adc_bool_idx = heuristics[1]
    is_dwi_bool_idx = heuristics[2]
    
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

    consensus_pred = po.DataFrame(
        np.concatenate(all_predictions_fold,axis=1))
    consensus_pred = consensus_pred.to_pandas().mode(1).iloc[:,0]
    consensus_pred = [x.upper() for x in consensus_pred]
    consensus_pred_heuristics = consensus_pred.copy()
    # apply heuristics
    consensus_pred_heuristics = np.where(
        is_t2w_bool_idx,"T2",consensus_pred_heuristics)
    consensus_pred_heuristics = np.where(
        is_adc_bool_idx,"ADC",consensus_pred_heuristics)
    consensus_pred_heuristics = np.where(
        is_dwi_bool_idx,"DWI",consensus_pred_heuristics)
    for se,st,pa,p,ph in zip(all_series_uid,all_study_uid,all_patient_id,
                             consensus_pred,consensus_pred_heuristics):
        print("{patient_id},{study_uid},{series_uid},{pred},{pred_heur}".format(
            patient_id=pa,study_uid=st,
            series_uid=se,pred=p,pred_heur=ph))