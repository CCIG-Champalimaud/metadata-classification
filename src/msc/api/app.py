import os
import time as time
import yaml
import dill
import numpy as np
import polars as pl
from dataclasses import dataclass
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from ..data_loading import read_data, summarise_columns
from ..entrypoints.predict import (
    predict_catboost,
    predict_non_catboost,
    get_consensus_predictions,
    get_average_predictions,
    apply_heuristics,
)
from ..heuristics import heuristics_dict
from .utils import DicomWebHelper, OrthancHelper

with open("config-api.yaml", "r") as o:
    configuration = yaml.safe_load(o)

ORTHANC_URL = os.environ.get("ORTHANC_URL", None)
ORTHANC_USER = os.environ.get("ORTHANC_USER", None)
ORTHANC_PASSWORD = os.environ.get("ORTHANC_PASSWORD", None)
DICOMWEB_URL = os.environ.get("DICOMWEB_URL", None)
DICOMWEB_USER = os.environ.get("DICOMWEB_USER", None)
DICOMWEB_PASSWORD = os.environ.get("DICOMWEB_PASSWORD", None)

if all([ORTHANC_URL is not None]):
    orthanc_helper = OrthancHelper(
        url=ORTHANC_URL, user=ORTHANC_USER, password=ORTHANC_PASSWORD
    )
else:
    orthanc_helper = None

if all([DICOMWEB_URL is not None]):
    dicomweb_helper = DicomWebHelper(
        user=DICOMWEB_USER,
        password=DICOMWEB_PASSWORD,
        url=DICOMWEB_URL,
    )
else:
    dicomweb_helper = None

app = FastAPI()


class PredictionRequest(BaseModel):
    """
    Request body for the prediction API.
    """

    model_id: str
    dicom_path: str


class OrthancPredictionRequest(BaseModel):
    """
    Request body for the Orthanc prediction API.
    """

    model_id: str
    study_uid: str
    update_labels: bool = False


class DICOMWebPredictionRequest(BaseModel):
    """
    Request body for the DICOM-web prediction API.
    """

    model_id: str
    study_uid: str


@dataclass
class ModelServer:
    model_dict: dict[str, str]
    matches: dict[str, list[str]] | None = None
    heuristics: dict[str, str] | None = None
    filters: dict[str, str | list[str]] | None = None

    def __post_init__(self):
        self.models = {
            k: dill.load(open(self.model_dict[k], "rb"))
            for k in self.model_dict
        }
        self.matches = {} if self.matches is None else self.matches
        self.heuristics = {} if self.heuristics is None else self.heuristics

        self.matches = {
            k: np.array(self.matches[k], dtype=str) for k in self.model_dict
        }

    def predict_from_features(self, model_id: str, features: pl.DataFrame):
        model = self.models[model_id]
        task = model["args"].get("task_name", "classification")
        is_catboost = model["args"].get("model_name") == "catboost"
        if is_catboost:
            curr_predictions = predict_catboost(model, features)
        else:
            curr_predictions = predict_non_catboost(
                model, features, match=self.matches.get(model_id)
            )

        if "patient_id" in features:
            patient_id = features["patient_id"]
        else:
            patient_id = features["study_uid"]

        if task == "classification":
            consensus_pred = get_consensus_predictions(curr_predictions)
            predictions_df = pl.from_dict(
                {
                    "patient_id": patient_id,
                    "study_uid": features["study_uid"],
                    "series_uid": features["series_uid"],
                    "prediction": consensus_pred,
                    "prediction_heuristics": consensus_pred,
                }
            )
            if self.heuristics.get(model_id) is not None:
                heuristics = heuristics_dict[self.heuristics.get(model_id)]
                if heuristics is not None:
                    heuristics_df = heuristics(features)
                    predictions = apply_heuristics(
                        predictions_df, heuristics_df
                    )
        if task == "regression":
            average_predictions = get_average_predictions(curr_predictions)
            predictions_df = pl.from_dict(
                {
                    "patient_id": patient_id,
                    "study_uid": features["study_uid"],
                    "series_uid": features["series_uid"],
                    "prediction": average_predictions,
                    "prediction_heuristics": average_predictions,
                }
            )
        predictions = predictions.to_dict(as_series=False)
        return predictions

    def predict(self, model_id: str, dicom_path: str):
        """
        Predict the type of sequence.
        """
        features = read_data(dicom_path)
        predictions = self.predict_from_features(model_id, features)
        return predictions

    def predict_api(self, prediction_request: PredictionRequest):
        """
        Predict the type of sequence.
        """
        start_time = time.time()
        prediction = self.predict(
            prediction_request.model_id, prediction_request.dicom_path
        )
        end_time = time.time()
        prediction["time"] = end_time - start_time
        return prediction

    def predict_orthanc_api(self, prediction_request: OrthancPredictionRequest):
        """
        Predict the types of sequences in an Orthanc study.
        """
        if orthanc_helper is None:
            raise HTTPException(
                status_code=400,
                detail="ORTHANC_URL, ORTHANC_USER and ORTHANC_PASSWORD should be defined.",
            )
        start_time = time.time()
        features = orthanc_helper.get_study_features(
            prediction_request.study_uid
        )
        features = summarise_columns(features)
        prediction = self.predict_from_features(
            prediction_request.model_id, features
        )
        end_time = time.time()
        prediction["time"] = end_time - start_time
        if prediction_request.update_labels:
            for series_uid, pred in zip(
                prediction["series_uid"], prediction["prediction_heuristics"]
            ):
                orthanc_helper.put_label("series", series_uid, pred)
            orthanc_helper.put_label(
                "studies",
                prediction_request.study_uid,
                f"Model_{prediction_request.model_id}",
            )
        return prediction

    def predict_dicomweb_api(
        self, prediction_request: DICOMWebPredictionRequest
    ):
        """
        Predict the types of sequences in an Orthanc study.
        """
        if dicomweb_helper is None:
            raise HTTPException(
                status_code=400,
                detail="DICOMWEB_URL, DICOMWEB_USER and DICOMWEB_PASSWORD should be defined.",
            )
        start_time = time.time()
        features = dicomweb_helper.get_study_features(
            prediction_request.study_uid
        )
        features = summarise_columns(features)
        prediction = self.predict_from_features(
            prediction_request.model_id, features
        )
        end_time = time.time()
        prediction["time"] = end_time - start_time
        return prediction


api_model_dict = {}
api_match_dict = {}
api_heuristics_dict = {}
api_filter_dict = {}
for model in configuration["models"]:
    api_model_dict[model] = configuration["models"][model]["path"]
    if "matches" in configuration["models"][model]:
        api_match_dict[model] = configuration["models"][model]["matches"]
    if "heuristics" in configuration["models"][model]:
        api_heuristics_dict[model] = configuration["models"][model][
            "heuristics"
        ]
    if "filters" in configuration["models"][model]:
        api_filter_dict[model] = configuration["models"][model]["filters"]

model_server = ModelServer(
    model_dict=api_model_dict,
    matches=api_match_dict,
    heuristics=api_heuristics_dict,
    filters=api_filter_dict,
)

app.add_api_route(
    "/predict", endpoint=model_server.predict_api, methods=["POST"]
)

app.add_api_route(
    "/predict-orthanc",
    endpoint=model_server.predict_orthanc_api,
    methods=["POST"],
)

app.add_api_route(
    "/predict-dicomweb",
    endpoint=model_server.predict_dicomweb_api,
    methods=["POST"],
)
