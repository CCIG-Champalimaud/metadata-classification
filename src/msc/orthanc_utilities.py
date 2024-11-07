from re import L
import requests
import polars as pl
from .dicom_feature_extraction import inverted_dicom_header_dict

ORTHANC_URL = "http://localhost:8042"


def get_studies() -> list[dict]:
    """
    Gets all studies from orthanc.

    Returns:
        list[dict]: studies.
    """
    return requests.get(
        f"{ORTHANC_URL}/studies",
        headers={"Accept": "application/json"},
    ).json()


def get_study(study_uid: str) -> dict:
    """
    Gets a study from orthanc.

    Args:
        study_uid (str): study uid.

    Returns:
        dict: study.
    """
    return requests.get(
        f"{ORTHANC_URL}/studies/{study_uid}",
        headers={"Accept": "application/json"},
    ).json()


def filter_on_main_dicom_tags(
    series: list[dict], filters: dict[str, str]
) -> list[dict]:
    """
    Filters series based on MainDicomTags.

    Args:
        study_series (list[dict]): series.
        filters (dict[str, str]): filters for MainDicomTags.

    Returns:
        list[dict]: series.
    """
    if filters is None:
        return True
    else:
        for k in filters:
            if isinstance(filters[k], list):
                if series["MainDicomTags"].get(k) not in filters[k]:
                    return False
            else:
                if series["MainDicomTags"].get(k) != filters[k]:
                    return False
    return True


def get_series_in_study(
    study_uid: str, filters: dict[str, str] | None = None
) -> dict:
    """
    Gets all series in a study from orthanc.

    Args:
        study_uid (str): study uid.
        filters (dict[str, str]): filters based on MainDicomTags.

    Returns:
        dict: study.
    """
    out = requests.get(
        f"{ORTHANC_URL}/studies/{study_uid}/series",
        headers={"Accept": "application/json"},
    ).json()
    out = [s for s in out if filter_on_main_dicom_tags(s, filters=filters)]
    return out


def get_series_metadata(series_uids: str) -> dict:
    """
    Gets metadata for a series from orthanc.

    Args:
        series_uid (str): series uid.

    Returns:
        dict: series.
    """
    return [
        requests.get(
            f"{ORTHANC_URL}/instances/{series_uid}/tags",
            headers={"Accept": "application/json"},
        ).json()
        for series_uid in series_uids
    ]


def process_series_metadata(metadata: dict[str, dict[str, str | int]]) -> dict:
    """
    Processes metadata for a series.

    Args:
        metadata (dict): metadata.

    Returns:
        dict: processed metadata.
    """
    output = {}
    for k in metadata:
        k_processed = tuple(k.split(","))
        if k_processed in inverted_dicom_header_dict:
            value = metadata[k]["Value"]
            value = "-" if value is None else value
            output[inverted_dicom_header_dict[k_processed]] = value
    missing = []
    for k in inverted_dicom_header_dict:
        if inverted_dicom_header_dict[k] not in output:
            missing.append(inverted_dicom_header_dict[k])
            output[inverted_dicom_header_dict[k]] = "-"
    print(output["diffusion_bvalue"])
    return output


def get_study_features(
    study_uid: str, filters: dict[str, str] | None = None
) -> pl.DataFrame:
    """
    Gets features for a study.

    Args:
        study_uid (str): study uid.

    Returns:
        pl.DataFrame: study features.
    """
    out = []
    series = get_series_in_study(study_uid, filters)
    for s in series:
        n = len(s["Instances"])
        metadata = get_series_metadata(s["Instances"])
        for m in metadata:
            m = process_series_metadata(m)
            m["patient_id"] = study_uid
            m["study_uid"] = study_uid
            m["series_uid"] = s["ID"]
            m["series_description"] = s["MainDicomTags"]["SeriesDescription"]
            m["number_of_images"] = n
            out.append(m)
    features = pl.DataFrame(out)
    return features


if __name__ == "__main__":
    all_study_uids = get_studies()

    for study_uid in all_study_uids:
        features = get_study_features(
            study_uid, {"BodyPartExamined": "PROSTATE", "Modality": "MR"}
        )
