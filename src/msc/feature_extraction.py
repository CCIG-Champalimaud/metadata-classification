import numpy as np
import polars as pl
from copy import deepcopy
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer


class SpaceSepNumColsToMatrix(BaseEstimator, TransformerMixin):
    """
    Converts a list of space-separated numbers into a matrix following a set
    of properties. This is useful for feature extraction when there is some
    structure to the input (i.e. each element is a set of numbers which can
    be summarised using relatively simple features).
    """

    def __init__(self, standard=False, default_value=-1, split_char: str = " "):
        """
        Args:
            standard (bool, optional): assumes that the same number of features
                is always present. If False, extracts the length, sum, minimum
                and maximum values from the space-separated array. Defaults to
                False.
            default_value (int, optional): default value for when standard is
                False. Defaults to -1.
            split_char (str, optional): character to split the space-separated
                numbers. Defaults to " ".
        """
        self.standard = standard
        self.default_value = default_value
        self.split_char = split_char

    def to_polars(self, X: np.ndarray | pl.DataFrame) -> pl.DataFrame:
        """
        Converts a numpy array or polars DataFrame to a polars DataFrame.

        Args:
            X (np.ndarray | pl.DataFrame): input data.

        Returns:
            pl.DataFrame: polars DataFrame.
        """
        if isinstance(X, pl.DataFrame) is False:
            X = pl.DataFrame({"feature": X})
        X.columns = ["feature"]
        return X

    def fit(self, X: np.ndarray, y: None = None) -> BaseEstimator:
        """
        Fits the transformer.

        Args:
            X (np.ndarray): input data.
            y (None, optional): not used. Defaults to None.
        """
        X = self.to_polars(X)
        sizes = X.with_columns(
            pl.col("feature").str.split(" ").list.len().alias("feature")
        )["feature"].to_list()
        if self.standard is False:
            self.transform_ = "sum_size"
            self.n_features_ = 5
            self.feature_names_ = ["length", "sum", "min", "max", "mean"]
        else:
            self.transform_ = "standard"
            self.n_features_ = sizes[0]
            self.feature_names_ = [i for i in range(self.n_features_)]
        return self

    def transform(self, X: np.ndarray, y: None = None) -> np.ndarray:
        """
        Transforms the input data containing space-separated columns into a
        numerical matrix.

        Args:
            X (np.ndarray): input data with strings containing space-separated
                columns.
            y (None, optional): not used. Defaults to None.

        Raises:
            Exception: raises an error if ``standard`` is ``True`` and there are
                a different number of elements in each row.

        Returns:
            np.ndarray: transformed data.
        """

        X = self.to_polars(X)
        if self.transform_ == "standard":
            X = np.array(
                X.with_columns(pl.col("feature").str.split(" ").explode())
            )
            mat = np.array(mat).astype(np.float32)
            if mat.shape[1] != self.n_features_:
                raise Exception("different number of elements")
        elif self.transform_ == "sum_size":
            mat = np.array(
                X.with_columns(
                    pl.col("feature")
                    .str.split(" ")
                    .alias("feature")
                    .list.eval(pl.element().cast(pl.Float32, strict=False))
                )
                .with_columns(
                    # ["length", "sum", "min", "max", "mean"]
                    pl.col("feature").list.len().alias("length"),
                    pl.col("feature").list.sum().alias("sum"),
                    pl.col("feature").list.min().alias("min"),
                    pl.col("feature").list.max().alias("max"),
                    pl.col("feature").list.mean().alias("mean"),
                )
                .drop("feature")
            )
            mat[np.isnan(mat)] = self.default_value
        return mat

    def fit_transform(self, X, y=None) -> np.ndarray:
        self.fit(X)
        return self.transform(X)


class TextColsToCounts(BaseEstimator, TransformerMixin):
    """
    Convenience class that converts text columns into a numerical matrix for
    regular machine-learning models. Columns in ``text_cols`` are converted
    to a matrix of counts using the standard ``sklearn`` CountVectorizer, while
    columns in ``text_num_cols`` are converted using
    ``SpaceSepNumColsToMatrix``. Finally, ``num_cols`` are simply coerced to
    numerical values.

    ``text_cols``, ``text_num_cols`` and ``num_cols`` have to be provided as
    dictionaries with the keys corresponding to array indices and the values to
    column names. This is helpful for when the data is loaded from a polars
    DataFrame.
    """

    def __init__(
        self,
        text_cols: dict[str, str] = {},
        text_num_cols: dict[str, str] = {},
        num_cols: dict[str, str] = {},
    ):
        """
        Args:
            text_cols (dict[str, str], optional): text columns. Defaults to {}.
            text_num_cols (dict[str, str], optional): numerical text columns.
                Defaults to {}.
            num_cols (dict[str, str], optional): numerical columns. Defaults to
                {}.
        """
        self.text_cols = text_cols
        self.text_num_cols = text_num_cols
        self.num_cols = num_cols

    def fit(self, X: np.ndarray, y: None = None) -> BaseEstimator:
        """
        Fits the transformer.

        Args:
            X (np.ndarray): input data array.
            y (None, optional): not used. Defaults to None.
        """
        self.all_cols_ = sorted(
            [*self.num_cols, *self.text_cols, *self.text_num_cols]
        )
        X = np.array(X)
        self.vectorizers_ = {}
        self.col_name_dict_ = {}
        for col in self.text_cols:
            self.vectorizers_[col] = CountVectorizer(
                stop_words=None,
                lowercase=True,
                analyzer="word",
                min_df=0.01,
                binary=True,
                ngram_range=(1, 5),
            )
            d = X[:, col]
            self.vectorizers_[col].fit(d)
            self.col_name_dict_[col] = [
                "{}:{}".format(self.text_cols[col], x)
                for x in self.vectorizers_[col].vocabulary_
            ]
        for col in self.text_num_cols:
            self.vectorizers_[col] = SpaceSepNumColsToMatrix()
            d = X[:, col]
            self.vectorizers_[col].fit(d)
            self.col_name_dict_[col] = [
                "{}:{}".format(self.text_num_cols[col], x)
                for x in self.vectorizers_[col].feature_names_
            ]
        for col in self.num_cols:
            self.col_name_dict_[col] = [self.num_cols[col]]
        self.new_col_names_ = []
        for c in self.col_name_dict_:
            self.new_col_names_.extend(self.col_name_dict_[c])
        return self

    def transform(
        self, X: np.ndarray | pl.DataFrame, y: None = None
    ) -> np.ndarray:
        """
        Transforms the input data containing text, text-numerical and numerical
        columns into a numerical matrix.

        Args:
            X (np.ndarray): input data array.
            y (None, optional): not used. Defaults to None.

        Returns:
            np.ndarray: transformed array.
        """
        if isinstance(X, pl.DataFrame):
            all_cols = [
                *[self.text_cols[k] for k in self.text_cols],
                *[self.text_num_cols[k] for k in self.text_num_cols],
                *[self.num_cols[k] for k in self.num_cols],
            ]
            text_cols = {v: k for k, v in self.text_cols.items()}
            text_num_cols = {v: k for k, v in self.text_num_cols.items()}
            num_cols = {v: k for k, v in self.num_cols.items()}
            output = []
            for col in all_cols:
                if col in text_cols:
                    mat = self.vectorizers_[text_cols[col]].transform(X[:, col])
                    mat = mat.todense()
                    t = "text"
                elif col in text_num_cols:
                    mat = self.vectorizers_[text_num_cols[col]].transform(
                        X[:, col]
                    )
                    t = "num_text"
                elif col in num_cols:
                    mat = np.array(X[:, col])[:, np.newaxis]
                    t = "num"
                output.append(np.array(mat))
        else:
            X = np.array(deepcopy(X))
            output = []
            for col in all_cols:
                if col in self.text_cols:
                    mat = self.vectorizers_[col].transform(X[:, col])
                    mat = mat.todense()
                    t = "text"
                elif col in self.text_num_cols:
                    mat = self.vectorizers_[col].transform(X[:, col])
                    t = "num_text"
                elif col in self.num_cols:
                    mat = np.array(X[:, col])[:, np.newaxis]
                    t = "num"
                else:
                    continue
                output.append(mat)
        output = np.concatenate(output, axis=1).astype(np.float32)
        return output

    def fit_transform(self, X, y=None):
        self.fit(X)
        return self.transform(X)


class RemoveNan(BaseEstimator, TransformerMixin):
    """
    Remove rows with more than 1 nan value. Not complicated.
    """

    def __init__(self):
        pass

    def fit(self, X: None = None, y: None = None) -> BaseEstimator:
        """
        Does nothing.

        Args:
            X (None): not used. Defaults to None.
            y (None): not used. Defaults to None.

        Returns:
            self
        """
        return self

    def transform(
        self, X: np.ndarray, y: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Drops rows with more than 1 nan value.

        Args:
            X (np.ndarray): data array.
            y (np.ndarray): target array.

        Returns:
            np.ndarray: ``X`` and ``y`` without rows with more than 1 nan value.
        """
        X = deepcopy(X)
        idxs = np.isnan(X).sum(1) > 1
        X = X[~idxs]
        y = y[~idxs]
        return X, y

    def fit_transform(self, X, y):
        return self.transform(X, y)
