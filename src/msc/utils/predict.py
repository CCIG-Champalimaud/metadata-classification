import argparse
import json
import dill
import numpy as np
import polars as pl
import os
from glob import glob
from multiprocessing import Pool
from scipy.stats import mode
from tqdm import tqdm
from catboost import FeaturesData
from ..dicom_feature_extraction import (
    extract_features_from_dicom,
    dicom_header_dict,
)
from ..feature_extraction import TextColsToCounts
from ..constants import text_sep_cols, num_sep_cols, num_cols
from ..sanitization import sanitize_data
from typing import Tuple, List


def read_data(
    input_paths: list[str], dicom_recursion: int = 0, n_workers: int = 0
) -> pl.DataFrame:
    def read_data_dicom_dataset(input_path, dicom_recursion, n_workers):
        all_series_paths = glob(
            os.sep.join(
                [
                    input_path.rstrip("/"),
                    *["*" for _ in range(dicom_recursion)],
                ]
            )
        )
        n = len(all_series_paths)
        features = {}
        if n_workers > 0:
            with Pool(n_workers) as p:
                for f in tqdm(
                    p.imap(extract_features_from_dicom, all_series_paths),
                    total=n,
                ):
                    for k in f:
                        if k not in features:
                            features[k] = []
                        features[k].append(f[k])
        else:
            for f in tqdm(
                map(extract_features_from_dicom, all_series_paths), total=n
            ):
                for k in f:
                    if k not in features:
                        features[k] = []
                    features[k].append(f[k])
        features = pl.from_dict(features)
        return features

    def read_data_dicom(input_path: str):
        features = extract_features_from_dicom(input_path)
        features = {k: [features[k]] for k in features}
        features = pl.from_dict(features)
        return features

    def read_data_csv_tsv(input_path: str):
        sep = "\t" if input_path[-3:] == "tsv" else ","
        features = pl.read_csv(input_path, sep=sep, ignore_errors=True)
        features.columns = [x.replace(" ", "_") for x in features.columns]
        return features

    def read_parquet(input_path: str):
        features = pl.read_parquet(input_path)
        features.columns = [x.replace(" ", "_") for x in features.columns]
        return features

    def read_json(input_path: str):
        json_file = json.load(open(input_path, "r"))
        output_dict = {"patient_id": [], "study_uid": [], "series_uid": []}
        for patient_id in json_file:
            for study_uid in json_file[patient_id]:
                for series_uid in json_file[patient_id][study_uid]:
                    md_dict = json_file[patient_id][study_uid][series_uid]
                    for metadata_key in md_dict:
                        if all(
                            [
                                metadata_key in dicom_header_dict,
                                metadata_key != "patient_id",
                                metadata_key != "study_uid",
                                metadata_key != "series_uid",
                            ]
                        ):
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
        features = pl.from_dict(output_dict)
        return features

    all_features = []
    for input_path in input_paths:
        extension = input_path.split(".")[-1]
        if os.path.isdir(input_path) == True:
            # load dicom metadata
            if dicom_recursion > 0:
                all_features.append(
                    read_data_dicom_dataset(
                        input_path, dicom_recursion, n_workers
                    )
                )
            else:
                all_features.append(read_data_dicom(input_path))
        elif extension in ["tsv", "csv"]:
            all_features.append(read_data_csv_tsv(input_path))
        elif extension == "parquet":
            all_features.append(read_parquet(input_path))
        elif extension == "json":
            all_features.append(read_json(input_path))
        else:
            raise NotImplementedError(
                "Input must be DICOM series dir or csv, tsv or parquet file"
            )
    if len(all_features) > 1:
        all_features = pl.concat(all_features, how="vertical")
    else:
        all_features = all_features[0]
    all_features = summarise_columns(all_features)
    return all_features


def summarise_columns(x: pl.DataFrame) -> pl.DataFrame:
    cols = x.columns
    group_cols = ["study_uid", "series_uid"]
    if "patient_id" in x:
        group_cols.append("patient_id")
    col_expressions = [
        (
            pl.col(k)
            .cast(pl.Utf8)
            .str.replace_all(r"^nan$|\\|;", "-")
            .str.replace_all(r"\\|;", " ")
            .str.replace_all(";", " ")
            .fill_null("-")
            .unique()
            .str.concat(" ")
            .alias(k)
        )
        for k in cols
        if (k not in group_cols) and (k in dicom_header_dict)
    ]
    if "number_of_images" in x:
        col_expressions.append(
            (pl.col("number_of_images").cast(pl.Int32).median())
        )
    elif "number_of_frames" in x:
        col_expressions.append(
            (
                pl.col("number_of_frames")
                .cast(pl.Int32)
                .median()
                .alias("number_of_images")
            )
        )
    elif "number_of_images" not in x:
        col_expressions.append(
            pl.col("study_uid").len().alias("number_of_images")
        )
    output = x.group_by(group_cols).agg(col_expressions)
    return output


def get_heuristics(
    features: pl.DataFrame,
) -> Tuple[List[bool], List[bool], List[bool]]:
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


def get_consensus_predictions(all_predictions: List[np.ndarray]) -> List[str]:
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

    # aggregate prediction consensus
    consensus_pred = get_consensus_predictions(all_predictions_fold)

    # calculate heuristics
    heuristics_df = get_heuristics(features)

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
