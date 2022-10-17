#!/usr/bin/env bash

python -m src.msc.utils.predict \
    --model_paths \
    models/catboost.percent_phase_field_of_view:sar:series_description.pkl \
    --input_path $1