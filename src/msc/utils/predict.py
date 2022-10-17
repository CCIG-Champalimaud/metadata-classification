import argparse
import dill
import numpy as np
import pandas as pd
from scipy import stats
from catboost import FeaturesData
from ..dicom_feature_extraction import extract_features_from_dicom
from ..constants import text_sep_cols,num_sep_cols,num_cols
from ..sanitization import sanitize_data

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Predicts the type of sequence.")
    
    parser.add_argument("--input_path",required=True,
                        help="Path to DICOM directory")
    parser.add_argument("--model_paths",required=False,nargs="+",
                        help="Path to model")

    args = parser.parse_args()
    
    # load dicom headers
    features = extract_features_from_dicom(args.input_path)
    features = pd.DataFrame.from_dict(
        {k:[features[k]] for k in features})
    features = sanitize_data(features)
    
    # setup models
    all_predictions = []
    match = ["ADC","DCE","DWI","Others","T2"]
    all_predictions_fold = []
    for model_path in args.model_paths:
        model = dill.load(open(model_path,"rb"))
        model_predictions = []
        for fold in model["cv"]:
            if fold["count_vec"] is not None:
                is_catboost = False
                count_vec = fold["count_vec"]
                transformed_features = count_vec.transform(features)
                prediction = fold["model"].predict(transformed_features)
                model_predictions.append(match[int(prediction)].upper())
                all_predictions_fold.append(match[int(prediction)].upper())
            else:
                is_catboost = True
                fc = text_sep_cols + num_sep_cols
                text_arr = []
                num_arr = []
                for f in fold["feature_names"]:
                    if f in fc:
                        text_arr.append(np.array(features[f]))
                    elif f in num_cols:
                        num_arr.append(np.array(features[f]))
                text_arr = np.array(text_arr).T
                num_arr = np.array(num_arr).T
                dat = FeaturesData(num_feature_data=num_arr.astype(np.float32),
                                   cat_feature_data=text_arr)
                prediction = fold["model"].predict(dat)
                model_predictions.append(prediction[0][0].upper())
                all_predictions_fold.append(prediction[0][0].upper())

        consensus_pred = pd.Series(model_predictions).mode()[0].upper()
        consensus_pred_fold = pd.Series(
            all_predictions_fold).mode()[0].upper()
        all_predictions.append(consensus_pred)
    print(",".join(all_predictions))