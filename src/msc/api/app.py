import os
import time as time
import yaml
from fastapi import FastAPI
from .utils import DicomWebHelper, OrthancHelper
from .app_funcs import ModelServer, configuration_to_model_dicts

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


api_model_dict, api_match_dict, api_heuristics_dict, api_filter_dict = (
    configuration_to_model_dicts(configuration)
)

model_server = ModelServer(
    model_dict=api_model_dict,
    matches=api_match_dict,
    heuristics=api_heuristics_dict,
    filters=api_filter_dict,
)
model_server.register_dicomweb_helper(dicomweb_helper)
model_server.register_orthanc_helper(orthanc_helper)

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
