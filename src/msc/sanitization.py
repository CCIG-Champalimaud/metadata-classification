import numpy as np
import polars as pl
from .constants import (
    text_sep_cols,
    num_sep_cols,
    num_cols,
    replace_cols,
    cols_to_drop,
)


rep_dict = {
    r"\|": " ",
    r"\-": " ",
    r";": " ",
    r",": " ",
    r"\.": " ",
    r"\(": " ",
    r"\)": " ",
    r"_": " ",
    r":": " ",
}


def sanitize_data(data):
    all_cols = list(set([*text_sep_cols, *num_sep_cols, *replace_cols]))
    col_expressions = []
    for col in all_cols:
        pl_col = pl.col(col)
        if col in text_sep_cols or col in num_sep_cols:
            for k in rep_dict:
                pl_col = pl_col.replace(k, rep_dict[k])
            pl_col = pl_col.str.strip_chars(" ")
        if col in replace_cols:
            pl_col = pl_col.replace("", replace_cols[col])
        pl_col = pl_col.str.to_lowercase()
        col_expressions.append(col)
    data = data.with_columns(col_expressions)
    return data


def data_loading_wraper(data_path, keep_series_uid: bool = False):
    # load data, fix some minor recurring issues
    data = pl.read_csv(data_path)
    data = data.with_columns(pl.col("class").str.to_lowercase().alias("class"))

    # sanitize data
    data = sanitize_data(data)

    cols_to_drop_current = list(cols_to_drop)
    if keep_series_uid:
        cols_to_drop_current.remove("series_uid")
    X = data.drop(cols_to_drop_current)
    y = data["class"].to_numpy()
    study_uids = data["study_uid"]
    unique_study_uids = list(set(study_uids))
    return X, y, study_uids, unique_study_uids
