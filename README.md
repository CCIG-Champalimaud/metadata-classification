# DICOM metadata series classification (MSC) mpMRI sequence classification from DICOM metadata

## Context

Sometimes, large collections of mpMRI studies have no annotations regarding the sequence type. Using relatively simple methods (extraction of DICOM metadata, CatBoost predictive modelling) we constructed here a classifier (final model in `models`) that can accurately perform this inference (results [here](https://bitbucket.org/fchampalimaud/dicom-msc/src/master/analysis/)). More concretely, this model retrieves the unique metadata values of a set of tags on each DICOM series and assigns one of the following sequence types to each series:

* T2-weighed imaging (T2)
* Apparent diffusion coefficient (ADC)
* Diffusion-weighted imaging (DWI)
* Dynamic contrast enhancement (DCE)
* Other sequence types (others)

## Conda environments

Conda environments are provided for both training and inference (`environment-docker.yaml` and `environment-docker-prediction-catboost.yaml`, respectively). We use micromamba as it is typically faster but whichever one the enduser prefers is fine.

## Code details

The Python code presented here concerns, loosely, feature extraction, model training and model inference.

### Feature extraction

*In `src/msc/dicom_feature_extraction.py`*

The extraction of metadata values from each DICOM series, which can be run as a command line utility - `python -m src.msc.dicom_feature_extraction path/to/dicom/series` - producing a JSON formatted dictionary where each value corresponds to the space-concatenated list of unique DICOM values for each tag.

### Model training

*In `src/msc/__main__.py`*

This is also a CLI (`python -m src.msc`) whose details are available with the `--help` flag. **Training requires as input a CSV file with all the tags produced by `src/msc/dicom_feature_extraction.py`.** Inspection of this code will reveal that there are several different possible models. A more practical example of training is available in the snakemake pipeline `train_all_models.smk`.

### Inference

*In `src/msc/utils/predict.py`*

This is what end users should focus on using - we have made it as simple as possible to use this, so it shouldn't take more than running the following code:

```bash
MODEL_PATH=models/catboost.percent_phase_field_of_view:sar:series_description.pkl
python src.msc.utils.predict \
    --input_path /path/to/input \
    --model_path $MODEL_PATH
```

or 

```bash
./predict.sh /path/to/input
```

The input path (`path/to/input`) can be:
* The path to a DICOM series folder
* A CSV, TSV or Parquet file with - at least - the columns which the model will be using and a `study_uid` and `series_uid` column. This file can either have one row for each instance (each field corresponding to one metadata value) or for each series (each field corresponds to the space-concantenated list of unique metadata values for this series), but the latter is preferred. **For reference:**
    * Predicting the sequence types for a TSV file with 5,520,218 instances (approximately 60K series) takes approximately 2 minutes and 30 seconds
    * Predicting the sequence types for a parquet file with 60K series (the same 60K series) takes approximately 30 seconds