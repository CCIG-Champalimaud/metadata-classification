import logging
import polars as pl
from .constants import (
    text_sep_cols,
    num_sep_cols,
    num_cols,
    replace_cols,
    cols_to_drop,
)


logger = logging.getLogger(__name__)


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


def sanitize_data(data: pl.DataFrame) -> pl.DataFrame:
    """
    Sanitizes the input DataFrame by performing various text and numeric
    transformations on the specified columns. Performs replacements according to
    ``rep_dict`` and replaces empty strings with the corresponding value in
    ``replace_cols``.

    Args:
        data (pl.DataFrame): the input DataFrame to be sanitized.

    Returns:
        pl.DataFrame: the sanitized DataFrame.
    """

    logger.info(
        "Sanitising data",
        extra={"n_rows": data.height, "n_columns": len(data.columns)},
    )
    all_cols = list(set([*text_sep_cols, *num_sep_cols, *replace_cols]))
    col_expressions = []
    for col in all_cols:
        if col not in data:
            continue
        pl_col = pl.col(col)
        if col in text_sep_cols or col in num_sep_cols:
            for k in rep_dict:
                pl_col = pl_col.replace(k, rep_dict[k])
            pl_col = pl_col.str.strip_chars(" ")
        if col in replace_cols:
            pl_col = pl_col.replace("", replace_cols[col])
        pl_col = pl_col.str.to_lowercase()
        col_expressions.append(pl_col.alias(col))
    if col_expressions:
        data = data.with_columns(col_expressions)
    logger.debug(
        "Finished sanitising data",
        extra={"n_rows": data.height, "n_columns": len(data.columns)},
    )
    return data
