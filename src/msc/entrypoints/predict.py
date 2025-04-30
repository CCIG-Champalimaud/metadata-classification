import argparse
import dill
import joblib
import numpy as np
import polars as pl
from scipy.stats import mode
from ..feature_extraction import TextColsToCounts
from ..constants import text_sep_cols, num_sep_cols, num_cols
from ..sanitization import sanitize_data
from ..data_loading import read_data
from ..heuristics import heuristics_dict


class RemappingUnpickler(dill.Unpickler):
    """Custom unpickler that remaps module paths during deserialization.

    This class extends dill.Unpickler to handle cases where package structures
    have changed between serialization and deserialization. It specifically
    remaps paths from "old_package" to "new_package" to maintain compatibility
    with previously serialized objects.

    Example:
        with open("model.pkl", "rb") as f:
            unpickler = RemappingUnpickler(f)
            model = unpickler.load()
    """

    def find_class(self, module, name):
        if module == "src.msc":
            module = "msc"
        elif module.startswith("src"):
            module = module.replace("src.", "", 1)
        return super().find_class(module, name)


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


def predict_non_catboost(
    model: dict, features: pl.DataFrame, match: np.ndarray = None
) -> list[np.ndarray]:
    curr_predictions = []
    for fold in model["cv"]:
        count_vec: TextColsToCounts = fold["count_vec"]
        count_vec.cols_to_drop = ["series_uid", "study_uid"]
        transformed_features = count_vec.transform(features)
        prediction = fold["model"].predict(transformed_features)
        if match is not None:
            curr_predictions.append(
                np.char.upper(match[prediction.astype(np.int32)])[:, None]
            )
        else:
            curr_predictions.append(prediction.astype(str)[:, None])
    return curr_predictions


def predict_catboost(model: dict, features: pl.DataFrame) -> list[np.ndarray]:
    from catboost import FeaturesData

    fc = text_sep_cols + num_sep_cols
    curr_predictions = []
    for fold in model["cv"]:
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
        curr_predictions.append(prediction.astype(str))
    return curr_predictions


def apply_heuristics(
    predictions_df: pl.DataFrame, heuristics_df: pl.DataFrame
) -> pl.DataFrame:
    feature_cols = [
        x for x in heuristics_df.columns if x not in ["study_uid", "series_uid"]
    ]
    heuristics_fn = pl.when(pl.col(feature_cols[0]) == True).then(
        pl.lit(feature_cols[0])
    )
    for x in feature_cols[1:]:
        heuristics_fn = heuristics_fn.when(pl.col(x) == True).then(pl.lit(x))
    heuristics_fn = heuristics_fn.otherwise(
        pl.col("prediction_heuristics")
    ).alias("prediction_heuristics")
    predictions_df = (
        predictions_df.join(heuristics_df, on=["study_uid", "series_uid"])
        .with_columns(heuristics_fn)
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
    return predictions_df


def main():
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
        "--heuristics",
        type=str,
        default=None,
        help="Heuristic function to apply to the data",
        choices=list(heuristics_dict.keys()),
    )
    parser.add_argument(
        "--classes",
        type=str,
        nargs="+",
        help="Sorted list of classes to predict",
    )
    parser.add_argument(
        "--reduce",
        default="series",
        type=str,
        help="Whether to reduce features to instance, series or study level",
        choices=["instance", "series", "study"],
    )
    parser.add_argument(
        "--n_workers",
        type=int,
        default=0,
        help="Number of workers when dicom_recursion > 0",
    )
    parser.add_argument(
        "--feature_column_mapping",
        type=str,
        default=None,
        nargs="+",
        help="List of input_path_names:model_name pairs for the data",
    )

    args = parser.parse_args()

    if args.reduce == "series":
        group_cols = ["study_uid", "series_uid", "patient_id"]
    elif args.reduce == "study":
        group_cols = ["study_uid", "patient_id"]
    elif args.reduce == "instance":
        group_cols = None
    features = read_data(
        args.input_paths,
        dicom_recursion=args.dicom_recursion,
        n_workers=args.n_workers,
        group_cols=group_cols,
        feature_column_mapping=args.feature_column_mapping,
    )
    features = sanitize_data(features)
    features = features.sort(by=["study_uid", "series_uid"])

    # setup models
    match = None
    if args.classes is not None:
        match = np.array(args.classes, dtype=str)
    all_predictions_fold = []

    # predict
    for model_path in args.model_paths:
        model_extension = model_path.split(".")[-1]
        if model_extension == "joblib":
            model = joblib.load(model_path)
        else:
            with open(model_path, "rb") as o:
                model = RemappingUnpickler(o).load()
        task = model["args"].get("task_name", "classification")
        is_catboost = model["args"].get("model_name") == "catboost"
        if is_catboost:
            curr_predictions = predict_catboost(model, features)
        else:
            curr_predictions = predict_non_catboost(model, features, match)
        all_predictions_fold.extend(curr_predictions)

    if "patient_id" in features:
        patient_id = features["patient_id"]
    else:
        patient_id = features["study_uid"]

    if task == "classification":
        # aggregate prediction consensus
        consensus_pred = get_consensus_predictions(all_predictions_fold)

        predictions_df = pl.from_dict(
            {
                "patient_id": patient_id,
                "study_uid": features["study_uid"],
                "series_uid": features["series_uid"],
                "prediction": consensus_pred,
                "prediction_heuristics": consensus_pred,
            }
        )

        # calculate heuristics
        if args.heuristics is not None:
            heuristics_df = heuristics_dict[args.heuristics](features)
            heuristics_df = apply_heuristics(predictions_df, heuristics_df)
    elif task == "regression":
        # aggregate prediction average
        predictions_df = get_average_predictions(all_predictions_fold)

    # print predctions
    print(predictions_df.to_pandas().to_csv(index=False))


if __name__ == "__main__":
    main()
