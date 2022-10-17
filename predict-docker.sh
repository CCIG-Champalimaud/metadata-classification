#!/usr/bin/env bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
/opt/conda/bin/python -m src.msc.utils.predict \
    --model_paths \
    models/catboost.percent_phase_field_of_view:sar:series_description.pkl \
    --input_path /dicom