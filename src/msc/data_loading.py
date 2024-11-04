import os
import json
import polars as pl
from glob import glob
from multiprocessing import Pool
from tqdm import tqdm
from .dicom_feature_extraction import (
    extract_features_from_dicom,
    image_feature_keys,
)
from .sanitization import sanitize_data
from .constants import cols_to_drop


def read_data_dicom_dataset(
    input_path: str, dicom_recursion: int, n_workers: int = 0
):
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
    features = pl.from_dict({k: features[k][0] for k in features})
    return features


def read_data_dicom(input_path: str):
    features = extract_features_from_dicom(input_path)
    features = {k: [features[k]] for k in features}
    features = pl.from_dict(features)
    return features


def read_data_csv_tsv(input_path: str):
    sep = "\t" if input_path[-3:] == "tsv" else ","
    features = pl.read_csv(input_path, separator=sep, ignore_errors=True)
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


def summarise_columns(
    x: pl.DataFrame,
    group_cols: list[str] = ["study_uid", "series_uid", "patient_id"],
) -> pl.DataFrame:
    """
    Summarises all columns in a dataframe by the grouping columns (typically
    "study_uid" and "series_uid").

    Args:
        x (pl.DataFrame): feature dataframe.
        group_cols (list[str], optional): columns for output grouping. Defaults
            to ["study_uid", "series_uid", "patient_id"].)

    Returns:
        pl.DataFrame: summarised column.
    """
    cols = x.columns
    group_cols = [x for x in group_cols if x in cols]
    if len(group_cols) == 0:
        raise ValueError("No column in group_cols was present in x")
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
        if (k not in group_cols)
        and (k not in ["number_of_images", "number_of_frames"])
        and k not in image_feature_keys
    ]
    numerical_col_expressions = [
        pl.col(k).cast(pl.Float32).mean()
        for k in cols
        if k in image_feature_keys
    ]
    col_expressions.extend(numerical_col_expressions)
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


def read_data(
    input_paths: list[str] | str,
    dicom_recursion: int = 0,
    n_workers: int = 0,
    group_cols: list[str] | None = ["study_uid", "series_uid", "patient_id"],
) -> pl.DataFrame:
    """
    Reads data which can be in multiple formats. The supported data formats are:
        - Folder containing DICOM (.dcm) files
        - Folder containing DICOM dataset in a nested structure (requires
            specifying the recursion depth, ``dicom_recursion``)
        - CSV/TSV/parquet file containing features (one column per feature with
            columns corresponding to study and series UID)
        - JSON file with a format such that:
        ```
            patient_id
            |-study_uid
            | |-series_uid
            | | |-file_path
            | | | |-feature_1: value
            | | | |-feature_2: value
            ...
        ```

    Args:
        input_paths (list[str] | str): input path or list of input paths.
        dicom_recursion (int, optional): number of directories to dive into when
            looking for a large structured DICOM dataset. If specified assumes
            that DICOM files are hidden in nested folders at the specified
            recursion depth. Defaults to 0.
        n_workers (int, optional): number of parallel workers when recursion is
            specified. Defaults to 0.
        group_cols (list[str] | None, optional): columns to use as groups when
            summarising columns. Defaults to ["study_uid", "series_uid",
            "patient_id"].

    Returns:
        pl.DataFrame: DICOM feature dataframe.
    """

    all_features = []
    if isinstance(input_paths, str):
        input_paths = [input_paths]
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
    if group_cols is not None:
        all_features = summarise_columns(all_features, group_cols)
    return all_features


def data_loading_wraper(
    data_path: str,
    keep_series_uid: bool = False,
    target_column: str = "class",
    task: str = "classification",
) -> pl.DataFrame:
    """
    Loads the data in a consistent and standardised way for training.

    Args:
        data_path (str): path to CSV file.
        keep_series_uid (bool, optional): keep series UID before returning.
            Defaults to False.
        target_column (str, optional): name of classification column. Defaults
            to "class".
        task (str, optional): name of task (can be either "classification" or
            "regression"). Defaults to "classification".

    Returns:
        pl.DataFrame: polars dataframe for training.
    """
    # load data, fix some minor recurring issues
    extension = data_path.split(".")[-1]
    if extension in ["tsv", "csv"]:
        data = read_data_csv_tsv(data_path)
    elif extension == "parquet":
        data = read_parquet(data_path)
    elif extension == "json":
        data = read_json(data_path)
    else:
        raise NotImplementedError("Input must be csv, tsv or parquet file")
    if task == "classification":
        data = data.with_columns(
            pl.col(target_column).str.to_lowercase().alias(target_column)
        )
    else:
        data = data.with_columns(
            pl.col(target_column).cast(pl.Float32, strict=False)
        ).filter(
            pl.col(target_column).is_not_null()
            & pl.col(target_column).is_finite()
        )

    # sanitize data
    data = sanitize_data(data)

    cols_to_drop_current = [x for x in cols_to_drop if x in data.columns]
    if target_column not in cols_to_drop_current:
        cols_to_drop_current.append(target_column)
    if keep_series_uid:
        cols_to_drop_current.remove("series_uid")
    X = data.drop(cols_to_drop_current)
    y = data[target_column].to_numpy()
    study_uids = data["study_uid"]
    unique_study_uids = list(set(study_uids))
    return X, y, study_uids, unique_study_uids
