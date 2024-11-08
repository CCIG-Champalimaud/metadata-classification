import requests
import polars as pl
from dataclasses import dataclass
from requests.auth import HTTPBasicAuth
from ..dicom_feature_extraction import (
    inverted_dicom_header_dict,
    process_series_description,
)


def process_value(value: str | list | float | int) -> str:
    """
    Processes a value.

    Args:
        value (str | list | float | int): value.

    Returns:
        str: processed value.
    """
    if isinstance(value, str):
        if "\\" in value:
            value = value.split("\\")
    if isinstance(value, (list, tuple)):
        return " ".join([str(x) for x in value])
    value = str(value)
    return value


@dataclass
class DicomWebHelper:
    """
    Helper class for dicom web.

    Raises:
        ValueError: _description_
    """

    user: str | None = None
    password: str | None = None
    url: str = "http://localhost:8042/dicom-web"

    STUDY_UID_KEY: str = "0020000D"
    SERIES_UID_KEY: str = "0020000E"
    INSTANCE_UID_KEY: str = "00080018"

    def __post_init__(self):
        if self.user is not None and self.password is not None:
            self.auth = HTTPBasicAuth(self.user, self.password)
        else:
            self.auth = None

        self.dicom_header_dict = {
            "".join(k): inverted_dicom_header_dict[k]
            for k in inverted_dicom_header_dict
        }

    def get(
        self,
        study_uid: str,
        series_uid: str | None = None,
        instance_uid: str | None = None,
    ) -> dict:
        """
        Gets all studies from dicom web.

        Args:
            study_uid (str): study uid.
            series_uid (uid): series uid.
            instance_uid (uid): instance uid.

        Returns:
            dict: study.
        """
        url = f"{self.url}/studies/{study_uid}/series"
        if series_uid is not None:
            url = f"{url}/{series_uid}/instances"
        if instance_uid is not None:
            url = f"{url}/{instance_uid}/metadata"
        return requests.get(
            url, auth=self.auth, headers={"Accept": "application/json"}
        ).json()

    def get_series_in_study(self, study_uid: str) -> dict:
        """
        Gets all series in a study from dicom web.

        Args:
            study_uid (str): study uid.

        Returns:
            dict: study.
        """
        return requests.get(
            f"{self.url}/studies/{study_uid}/series",
            auth=self.auth,
            headers={"Accept": "application/json"},
        ).json()

    def get_series_metadata(self, study_uid: str, series_uid: str) -> dict:
        """
        Gets metadata for a series from orthanc.

        Args:
            study_uid (str): study uid.
            series_uid (str): series uid.

        Returns:
            dict: series.
        """
        instance_uids = self.get(study_uid, series_uid)
        return [
            self.get(
                study_uid,
                series_uid,
                instance_uid[self.INSTANCE_UID_KEY]["Value"][0],
            )[0]
            for instance_uid in instance_uids
        ]

    def process_series_metadata(
        self, metadata: dict[str, dict[str, str | int]]
    ) -> dict:
        """
        Processes metadata for a series.

        Args:
            metadata (dict): metadata.

        Returns:
            dict: processed metadata.
        """
        output = {}
        for k in metadata:
            if k in self.dicom_header_dict:
                value = metadata[k]["Value"]
                value = "-" if value is None else process_value(value)
                output[self.dicom_header_dict[k]] = value
        missing = []
        for k in self.dicom_header_dict:
            if self.dicom_header_dict[k] not in output:
                missing.append(self.dicom_header_dict[k])
                output[self.dicom_header_dict[k]] = "-"
        return output

    def get_study_features(self, study_uid: str) -> pl.DataFrame:
        """
        Gets features for a study.

        Args:
            study_uid (str): study uid.

        Returns:
            pl.DataFrame: study features.
        """
        out = []
        series = self.get(study_uid)
        for s in series:
            metadata = self.get_series_metadata(
                study_uid, s[self.SERIES_UID_KEY]["Value"][0]
            )
            n = len(metadata)
            for m in metadata:
                m = self.process_series_metadata(m)
                m["patient_id"] = study_uid
                m["study_uid"] = study_uid
                m["series_uid"] = s[self.SERIES_UID_KEY]["Value"][0]
                m["number_of_images"] = len(metadata)
                m["series_description"] = process_series_description(
                    m["series_description"]
                )
                out.append(m)
        features = pl.DataFrame(out)
        return features


@dataclass
class OrthancHelper:
    """
    Helper class for orthanc.
    """

    user: str | None = None
    password: str | None = None
    url: str = "http://localhost:8042"

    def __post_init__(self):
        if self.user is not None and self.password is not None:
            self.auth = HTTPBasicAuth(self.user, self.password)
        else:
            self.auth = None

    def get_studies(self) -> list[dict]:
        """
        Gets all studies from orthanc.

        Returns:
            list[dict]: studies.
        """
        return requests.get(
            f"{self.url}/studies",
            auth=self.auth,
            headers={"Accept": "application/json"},
        ).json()

    def get_study(self, study_uid: str) -> dict:
        """
        Gets a study from orthanc.

        Args:
            study_uid (str): study uid.

        Returns:
            dict: study.
        """
        return requests.get(
            f"{self.url}/studies/{study_uid}",
            auth=self.auth,
            headers={"Accept": "application/json"},
        ).json()

    def filter_on_main_dicom_tags(
        self, series: list[dict], filters: dict[str, str]
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
        self, study_uid: str, filters: dict[str, str] | None = None
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
            f"{self.url}/studies/{study_uid}/series",
            auth=self.auth,
            headers={"Accept": "application/json"},
        ).json()
        out = [
            s for s in out if self.filter_on_main_dicom_tags(s, filters=filters)
        ]
        return out

    def get_series_metadata(self, series_uids: str) -> dict:
        """
        Gets metadata for a series from orthanc.

        Args:
            series_uid (str): series uid.

        Returns:
            dict: series.
        """
        return [
            requests.get(
                f"{self.url}/instances/{series_uid}/tags",
                auth=self.auth,
                headers={"Accept": "application/json"},
            ).json()
            for series_uid in series_uids
        ]

    def process_series_metadata(
        self, metadata: dict[str, dict[str, str | int]]
    ) -> dict:
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
                value = "-" if value is None else process_value(value)
                output[inverted_dicom_header_dict[k_processed]] = value
        missing = []
        for k in inverted_dicom_header_dict:
            if inverted_dicom_header_dict[k] not in output:
                missing.append(inverted_dicom_header_dict[k])
                output[inverted_dicom_header_dict[k]] = "-"
        return output

    def get_study_features(
        self, study_uid: str, filters: dict[str, str] | None = None
    ) -> pl.DataFrame:
        """
        Gets features for a study.

        Args:
            study_uid (str): study uid.

        Returns:
            pl.DataFrame: study features.
        """
        out = []
        series = self.get_series_in_study(study_uid, filters)
        for s in series:
            n = len(s["Instances"])
            metadata = self.get_series_metadata(s["Instances"])
            for m in metadata:
                m = self.process_series_metadata(m)
                m["patient_id"] = study_uid
                m["study_uid"] = study_uid
                m["series_uid"] = s["ID"]
                m["series_description"] = process_series_description(
                    s["MainDicomTags"]["SeriesDescription"]
                )
                m["number_of_images"] = n
                out.append(m)
        features = pl.DataFrame(out)
        return features

    def put_label(self, category: str, identifier: str, label: str) -> None:
        """
        Puts a label on a study, series or patient.

        Args:
            category (str): category of the label.
            identifier (str): identifier of the label.
        """
        valid_categories = ["patients", "studies", "series"]
        if category not in valid_categories:
            raise ValueError(f"Category must be one of {valid_categories}")
        requests.put(
            f"{self.url}/{category}/{identifier}/labels/{label}",
            headers={"Accept": "application/json"},
        )
