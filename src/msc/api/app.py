import os
import time as time
import logging
import yaml
from fastapi import FastAPI
from .utils import DicomWebHelper, OrthancHelper
from .app_funcs import ModelServer, configuration_to_model_dicts


logger = logging.getLogger(__name__)


with open("config-api.yaml", "r") as o:
    configuration = yaml.safe_load(o)
logger.info(
    "Loaded API configuration",
    extra={"n_models": len(configuration.get("models", {}))},
)

ORTHANC_URL = os.environ.get("ORTHANC_URL", None)
ORTHANC_USER = os.environ.get("ORTHANC_USER", None)
ORTHANC_PASSWORD = os.environ.get("ORTHANC_PASSWORD", None)
DICOMWEB_URL = os.environ.get("DICOMWEB_URL", None)
DICOMWEB_USER = os.environ.get("DICOMWEB_USER", None)
DICOMWEB_PASSWORD = os.environ.get("DICOMWEB_PASSWORD", None)

if all([ORTHANC_URL is not None]):
    logger.info(
        "Initializing Orthanc helper",
        extra={"url": ORTHANC_URL, "user_set": ORTHANC_USER is not None},
    )
    orthanc_helper = OrthancHelper(
        url=ORTHANC_URL, user=ORTHANC_USER, password=ORTHANC_PASSWORD
    )
else:
    logger.warning(
        "ORTHANC configuration not set; Orthanc integration disabled"
    )
    orthanc_helper = None

if all([DICOMWEB_URL is not None]):
    logger.info(
        "Initializing DICOMweb helper",
        extra={"url": DICOMWEB_URL, "user_set": DICOMWEB_USER is not None},
    )
    dicomweb_helper = DicomWebHelper(
        user=DICOMWEB_USER,
        password=DICOMWEB_PASSWORD,
        url=DICOMWEB_URL,
    )
else:
    logger.warning(
        "DICOMweb configuration not set; DICOMweb integration disabled"
    )
    dicomweb_helper = None

app = FastAPI()


api_model_dict, api_match_dict, api_heuristics_dict, api_filter_dict = (
    configuration_to_model_dicts(configuration)
)
logger.info(
    "Configured model server",
    extra={
        "n_models": len(api_model_dict),
        "has_matches": bool(api_match_dict),
        "has_heuristics": bool(api_heuristics_dict),
        "has_filters": bool(api_filter_dict),
    },
)

model_server = ModelServer(
    model_dict=api_model_dict,
    matches=api_match_dict,
    heuristics=api_heuristics_dict,
    filters=api_filter_dict,
)
model_server.register_dicomweb_helper(dicomweb_helper)
model_server.register_orthanc_helper(orthanc_helper)
logger.info("Registered helpers with model server")

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
logger.info(
    "API routes registered",
    extra={"routes": ["/predict", "/predict-orthanc", "/predict-dicomweb"]},
)
