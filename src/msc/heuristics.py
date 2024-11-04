import numpy as np
import polars as pl


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
            "T2W": is_t2w_bool_idx,
            "ADC": is_adc_bool_idx,
            "DWI": is_dwi_bool_idx,
        }
    )
    return heuristics_df


heuristics_dict = {"prostate_mpmri": get_heuristics_prostate_mpmri}
