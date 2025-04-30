import numpy as np
import polars as pl
from dataclasses import dataclass


@dataclass
class StringVariable(str):
    value: str

    def __post_init__(self):
        self.value = self.value.lower()

    def contains_generic(
        self, value: list[str], list_of_substr: list[str] | str
    ):
        return [
            (substr in value if substr[0] != "~" else substr[1:] not in value)
            for substr in list_of_substr
        ]

    def contains_all(self, value: list[str], list_of_substr: list[str]):
        return all(self.contains_generic(value, list_of_substr))

    def contains_any(self, value: list[str], list_of_substr: list[str]):
        return any(self.contains_generic(value, list_of_substr))

    def contains(
        self, list_of_substr: list[str], check: str = "all", split: str = None
    ) -> bool:
        # coherce to string
        if isinstance(list_of_substr, str):
            list_of_substr = [list_of_substr]
        if check is not None:
            value = self.value.split(split)
        if check == "all":
            return self.contains_all(value, list_of_substr)
        elif check == "any":
            return self.contains_any(value, list_of_substr)


def get_bvalues(features: pl.DataFrame) -> list[int | None | str]:
    """
    Wrapper function to extract b-values from images.

    Args:
        features (pl.DataFrame): data frame containing "diffusion_bvalue",
            "diffusion_bvalue_ge" or "diffusion_bvalue_siemens" columns.

    Returns:
        list[int | None | str]: list of values corresponding to bvalues, one
            for each row.
    """
    bvalues = features["diffusion_bvalue"].to_list()
    for key in ["diffusion_bvalue_ge", "diffusion_bvalue_siemens"]:
        if key in features:
            bvalues_proxy = features[key].to_list()
            for i in range(len(bvalues)):
                if bvalues[i] == "-" and bvalues_proxy[i] != "-":
                    bvalues[i] = bvalues_proxy[i]
    return bvalues


def get_heuristics_prostate_mpmri(
    features: pl.DataFrame,
) -> tuple[list[bool], list[bool], list[bool]]:
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
    is_series_bool_idx = {
        "DWI": [],
        "T2W": [],
        "ADC": [],
    }
    image_types = features["image_type"].to_list()
    series_descriptions = features["series_description"].to_list()
    bvalues = get_bvalues(features)
    for it, sd, f in zip(image_types, series_descriptions, bvalues):
        sd = StringVariable(sd)
        it = StringVariable(it)
        f = [
            float(x) if x.replace(".", "").isnumeric() else 0
            for x in np.unique(f.split())
        ]
        if len(f) == 0:
            f = 0
        f = np.max(np.int32(f))
        # check if it can be t2w ("t2" substring in sd AND no
        # "cor" or "sag" substring)
        if sd.contains(["t2", "~cor", "~sag"]):
            is_series_bool_idx["T2W"].append(True)
        else:
            is_series_bool_idx["T2W"].append(False)
        # check if it can be dwi by seeing whether the maximum
        # b-value is greater than 0 and that "adc" is not in the
        # series description or image type
        if all([f > 0, sd.contains("~adc"), it.contains("~adc")]):
            is_series_bool_idx["DWI"].append(True)
        else:
            is_series_bool_idx["DWI"].append(False)
        # check if it can be adc ("adc" substring in it)
        if any([("adc" in it), sd.contains("adc", split=" ")]):
            is_series_bool_idx["ADC"].append(True)
        else:
            is_series_bool_idx["ADC"].append(False)
    heuristics_df = pl.DataFrame(
        {
            "study_uid": features["study_uid"],
            "series_uid": features["series_uid"],
            **is_series_bool_idx,
        }
    )
    return heuristics_df


heuristics_dict = {"prostate_mpmri": get_heuristics_prostate_mpmri}
