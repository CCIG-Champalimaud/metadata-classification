import yaml
from msc.api.utils import DicomWebHelper
from ..api.app_funcs import (
    configuration_to_model_dicts,
    ModelServer,
    DICOMWebPredictionRequest,
)


def main():
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dicomweb_url",
        type=str,
        help="DICOMweb URL",
        required=True,
    )
    parser.add_argument(
        "--dicomweb_user",
        type=str,
        help="DICOMweb user",
        default=None,
    )
    parser.add_argument(
        "--dicomweb_password",
        type=str,
        help="DICOMweb password",
        default=None,
    )
    parser.add_argument(
        "--model_config",
        type=str,
        help="Model configuration YAML",
        default="config-api.yaml",
    )
    parser.add_argument(
        "--model_id",
        type=str,
        help="Model identifier as specified in --model_config",
        required=True,
    )
    parser.add_argument(
        "--study_uid",
        type=str,
        help="Study UID",
        required=True,
        nargs="+",
    )

    args = parser.parse_args()

    with open(args.model_config, "r") as o:
        configuration = yaml.safe_load(o)

    dicomweb_helper = DicomWebHelper(
        url=args.dicomweb_url,
        user=args.dicomweb_user,
        password=args.dicomweb_password,
    )
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
    prediction_request = DICOMWebPredictionRequest(
        prediction_model_id=args.model_id,
        study_uid=args.study_uid,
        dicom_web_url=args.dicomweb_url,
    )

    return model_server.predict_dicomweb_api(prediction_request)


if __name__ == "__main___":
    main()
