import argparse
import dill
import numpy as np
import polars as pl
from scipy.stats import mode
from catboost import FeaturesData
from tqdm import tqdm
from typing import Tuple, list
from ..dicom_feature_extraction import dicom_header_dict
from ..feature_extraction import TextColsToCounts
from ..constants import text_sep_cols, num_sep_cols, num_cols
from ..sanitization import sanitize_data
from ..data_loading import read_data


def get_heuristics(
    features: pl.DataFrame,
) -> Tuple[list[bool], list[bool], list[bool]]:
    """
    Gets classification heuristics from feature df.

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
    image_types = features["image_type"].to_list()
    series_descriptions = features["series_description"].to_list()
    bvalues = features["diffusion_bvalue"].to_list()
    for key in ["diffusion_bvalue_ge", "diffusion_bvalue_siemens"]:
        if key in features:
            bvalues_proxy = features[key].to_list()
            for i in range(len(bvalues)):
                if bvalues[i] == "-" and bvalues_proxy[i] != "-":
                    bvalues[i] = bvalues_proxy[i]
    for it, sd, f in zip(image_types, series_descriptions, bvalues):
        it = it.lower()
        sd = sd.lower()
        f = [
            float(x) if x.replace(".", "").isnumeric() else 0
            for x in np.unique(f.split())
        ]
        if len(f) == 0:
            f = 0
        f = np.max(np.int32(f))
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
    heuristics_df = pl.DataFrame(
        {
            "study_uid": features["study_uid"],
            "series_uid": features["series_uid"],
            "t2w_heuristics": is_t2w_bool_idx,
            "adc_heuristics": is_adc_bool_idx,
            "dwi_heuristics": is_dwi_bool_idx,
        }
    )
    return heuristics_df


def mode(x: np.ndarray) -> np.ndarray:
    """Calculates the mode of an array.

    Args:
        x (np.ndarray): array to calculate the mode.
    Returns:
        np.ndarray: mode of the array.
    """
    u, c = np.unique(x, return_counts=True)
    return u[c.argmax()]


def get_consensus_predictions(all_predictions: list[np.ndarray]) -> list[str]:
    """Calculates consensus predictions (mode) from a list of prediction
    vectors.

    Args:
        all_predictions (List[np.ndarray]): list of prediction vectors.

    Returns:
        List[str]: consensus predictions.
    """
    consensus_pred = np.concatenate(all_predictions, axis=1)
    consensus_pred = [mode(x).upper() for x in consensus_pred]
    return consensus_pred


def get_average_predictions(all_predictions: list[np.ndarray]) -> list[str]:
    """Calculates average from a list of prediction vectors.

    Args:
        all_predictions (List[np.ndarray]): list of prediction vectors.

    Returns:
        List[str]: average predictions.
    """
    consensus_pred = (
        np.concatenate(all_predictions, axis=1).mean(axis=1).tolist()
    )
    return consensus_pred


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Predicts the type of sequence."
    )

    parser.add_argument(
        "--input_paths",
        required=True,
        nargs="+",
        help="Path to DICOM directory or CSV/TSV containing data",
    )
    parser.add_argument(
        "--model_paths", required=False, nargs="+", help="Path to model"
    )
    parser.add_argument(
        "--dicom_recursion",
        type=int,
        default=0,
        help="How deep in the folder structure are DICOM series",
    )
    parser.add_argument(
        "--n_workers",
        type=int,
        default=0,
        help="Number of workers when dicom_recursion > 0",
    )

    args = parser.parse_args()

    # if it's a directory, assume it's a directory containing .dcm files
    features = read_data(
        args.input_paths,
        dicom_recursion=args.dicom_recursion,
        n_workers=args.n_workers,
    )
    features = sanitize_data(features)
    features = features.sort(by=["study_uid", "series_uid"])

    # setup models
    all_predictions = []
    match = np.array(["ADC", "DCE", "DWI", "Others", "T2"], dtype=str)
    all_predictions_fold = []
    all_series_uid = features["series_uid"]
    all_study_uid = features["study_uid"]
    if "patient_id" in features:
        all_patient_id = features["patient_id"]
    else:
        all_patient_id = features["study_uid"]

    # predict
    for model_path in args.model_paths:
        model = dill.load(open(model_path, "rb"))
        task = model["args"].get("task_name", "classification")
        for fold in model["cv"]:
            if fold["count_vec"] is not None:
                is_catboost = False
                count_vec: TextColsToCounts = fold["count_vec"]
                count_vec.cols_to_drop = ["series_uid", "study_uid"]
                transformed_features = count_vec.transform(features)
                prediction = fold["model"].predict(transformed_features)
                all_predictions_fold.append(
                    np.char.upper(match[prediction.astype(np.int32)])[:, None]
                )
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
                dat = FeaturesData(
                    num_feature_data=num_arr.astype(np.float32),
                    cat_feature_data=text_arr,
                )
                prediction = fold["model"].predict(dat)
                all_predictions_fold.append(prediction.astype(str))

    if task == "classification":
        # aggregate prediction consensus
        consensus_pred = get_consensus_predictions(all_predictions_fold)

        # calculate heuristics
        heuristics_df = get_heuristics(features)

    elif task == "regression":
        # aggregate prediction average
        consensus_pred = get_average_predictions(all_predictions_fold)

    # merge predictions with heuristics
    if "patient_id" in features:
        patient_id = features["patient_id"]
    else:
        patient_id = features["study_uid"]
    predictions_df = (
        pl.from_dict(
            {
                "patient_id": patient_id,
                "study_uid": features["study_uid"],
                "series_uid": features["series_uid"],
                "prediction": consensus_pred,
            }
        )
        .join(heuristics_df, on=["study_uid", "series_uid"])
        .with_columns(pl.col("prediction").alias("prediction_heuristics"))
        .with_columns(
            pl.when(pl.col("t2w_heuristics") == True)
            .then(pl.lit("T2"))
            .when(pl.col("dwi_heuristics") == True)
            .then(pl.lit("DWI"))
            .when(pl.col("adc_heuristics") == True)
            .then(pl.lit("ADC"))
            .otherwise(pl.col("prediction_heuristics"))
            .alias("prediction_heuristics")
        )
        .select(
            [
                "patient_id",
                "study_uid",
                "series_uid",
                "prediction",
                "prediction_heuristics",
            ]
        )
    )

    # print predctions
    print(predictions_df.to_pandas().to_csv(index=False))
