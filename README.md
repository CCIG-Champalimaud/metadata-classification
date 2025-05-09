# DICOM metadata series classification (MSC) mpMRI sequence classification from DICOM metadata

## Context

Sometimes, large collections of mpMRI studies have no annotations regarding the sequence type. Using relatively simple methods (extraction of DICOM metadata, CatBoost predictive modelling) we constructed here a classifier (final model in `models`) that can accurately perform this inference. More concretely, this model retrieves the unique metadata values of a set of tags on each DICOM series and assigns one of the following sequence types to each series:

* T2-weighed imaging (T2)
* Apparent diffusion coefficient (ADC)
* Diffusion-weighted imaging (DWI)
* Dynamic contrast enhancement (DCE)
* Other sequence types (others)

## Environments and installation

To manage dependencies we use `micromamba` and `uv`, a Python package manager. To install `uv` and all the dependencies, run:

1. `micromamba create -n dicom-classification`
2. `micromamba activate dicom-classification`
3. `micromamba install python=$(cat .python-version) uv -c conda-forge -y`
4. `uv sync`
5. `uv pip install -r pyproject.toml`

And you should be all set!

### Optional dependencies

In `pyproject.toml` we specify different optional dependencies depending on what you want to do. This helps us keep the dependencies manageable and light.

#### Model training 

Add `--extra train` to `uv sync`.

#### Model serving

If you want to serve models using `fastapi`, you need to add `--extra serve` to `uv sync`.

#### Model serving with `catboost`, `lightgbm` or `xgboost`

If you want to serve `catboost`, `lightgbm` or `xgboost` models, you need to add `--extra server catboost`, `--extra server lightgbm` or `--extra server xgboost` to `uv sync`, respectively.

#### Pixel information extraction

If you want to extract pixel features you need to add `--extra pixel` to `uv sync`.

### Package building

You can also build the package using `uv` using `uv build`.

## Code details

The Python code presented here concerns, loosely, feature extraction, model training and model inference.

### Feature extraction

*In `src/msc/get_feature_parquet.py`*

The extraction of metadata values from each DICOM series, which can be run as a command line utility - `msc_feature_extraction --input_dir <PATH TO DICOM DATASET> --pattern '*dcm' --n_workers 8 --output_path <OUTPUT PATH>` - producing a parquet file with all the metadata values for each DICOM series. It is possible to extract two feature types and these are specified using `--feature_types`:

* `metadata`: extracts standard DICOM metadata values
* `image` extracts features retrieved from the pixel array of each individual series

### Model training

*In `src/msc/entrypoints/train.py`*

This is also a CLI (`msc_train`) whose details are available with the `--help` flag. **Training requires as input a CSV, TSV or parquet file with all the tags produced by `msc_feature_extraction`.** Inspection of this code will reveal that there are several different possible models. A more practical example of training is available in the snakemake pipeline `train_all_models.smk`.

### Inference (prediction)

*In `src/msc/entrypoints/predict.py`*

This is what end users should focus on using - we have made it as simple as possible to use this, so it shouldn't take more than running the following code:

```bash
# this requires training a model
MODEL_PATH=models/catboost.percent_phase_field_of_view:sar:series_description.pkl
python msc_predict \
    --input_path <PATH TO DICOM DIRECTORY> \
    --model_path $MODEL_PATH
```

The input path (`<PATH TO DICOM DIRECTORY>`) can be:
* The path to a DICOM series folder
* The path to a DICOM dataset (or study). For this, `--dicom_recursion` has to be specified to how deep in the folder structure one is expected to find DICOM files.
* A CSV, TSV or Parquet file with - at least - the columns which the model will be using and a `study_uid` and `series_uid` column. This file can either have one row for each instance (each field corresponding to one metadata value) or for each series (each field corresponds to the space-concantenated list of unique metadata values for this series), but the latter is preferred. **For reference:**
    * Predicting the sequence types for a TSV file with 5,520,218 instances (approximately 60K series) takes approximately 2 minutes and 30 seconds
    * Predicting the sequence types for a parquet file with 60K series (the same 60K series) takes approximately 30 seconds

#### Heuristics

Inference can apply heuristics on top of the predictions to obtain refined predictions. By heuristics, we mean a set of hard-coded rules that are used to determine the sequence type of a DICOM series. These are defined in `src/msc/heuristics.py` and are used to determine the sequence type of a DICOM series and override the prediction. The output of each heuristic should be a `polars` dataframe with a `study_uid` column, a `series_uid` column and a set of columns with names equal to the possible classifications. If the heuristics column for a given class is True, this class overrides the prediction. Please note that the last heuristic in the list overrides the formers, so the order of the heuristics is important.

### Model serving through APIs

*In `src/msc/api/app.py`*

To perform model serving with a dedicated API, we use FastAPI. So running this is as simple as running `fastapi run src/msc/api/app.py` (or `fastapi dev src/msc/api/app.py` for development). Inferences are supported for:

* **Standard queries**: a post request to `http://localhost:8000/predict` with a JSON body with the following keys:
    * `dicom_path`: the path to the DICOM directory (will recursively fetch all DICOM series)
    * `model_id`: the name of the model to use (see below how this is configured)
* **Orthanc queries**: a post request to `http://localhost:8000/predict-orthanc` with a JSON body with the following keys (this does not support image features):
    * `study_uid`: study UID in Orthanc
    * `model_id`: the name of the model to use (see below how this is configured)
* **DICOM-web queries**: a post request to `http://localhost:8000/predict-dicomweb` with a JSON body with the following keys (this does not support image features):
    * `study_uid`: study UID in DICOM-web
    * `model_id`: the name of the model to use (see below how this is configured)

#### Configuration

The configuration for the API is available in `config-api.yaml` and `src/msc/api/app.py` automatically fetches this file. Currently, this makes use of four possible values for each model (specified under a `models` key):

* `model_dict`: a dictionary with keys being the names of the models and values being the paths to the models
* `matches`: a dictionary with keys being the names of the models and values being the list of sequence types the model can predict (this is only used when the model is not a CatBoost model)
* `heuristics`: a dictionary with keys being the names of the models and values being the heuristics used to determine the sequence type. Heuristics are defined in `src/msc/heuristics.py` (more information below)
* `filters`: a dictionary with keys being the names of the models and values being a dictionary with keys being the DICOM tags and values being a list of values. These are used to filter the DICOM series before prediction and only works when using the Orthanc prediction API.

Example of `config-api.yaml`: 

```
models:
  prostate_mpmri:
    model: "/storage/models/metadata/models/xgb.standard.100.pkl"
    matches: ["ADC", "DCE", "DWI", "Others", "T2W"]
    heuristics: "prostate_mpmri"
    filters:
      BodyPartExamined: ["PROSTATE", "PELVIS", "GENITOURINARY SYSTEM"]
      Modality: ["MR"]
```

#### Defining URLs and users for Orthanc and DICOM-web

The following environment variables are used to define the URLs and users for Orthanc and DICOM-web:

* `ORTHANC_URL`: the URL of the Orthanc server
* `ORTHANC_USER`: the username for the Orthanc server
* `ORTHANC_PASSWORD`: the password for the Orthanc server
* `DICOMWEB_URL`: the URL of the DICOM-web server
* `DICOMWEB_USER`: the username for the DICOM-web server
* `DICOMWEB_PASSWORD`: the password for the DICOM-web server

We recommend having a file (i.e. `.env`) which is sourced with these constants prior to API utilization as it greatly simplifies this process.

While we considered including the Orthanc and DICOM-web credentials in the configuration file, we decided against this as it would make it easier to leak the credentials (and this allows easier deployment through Docker/Docker-compose/Kubernetes/etc through the definition of environment variables in the manifest files).
