import os
import re
import pydicom
import numpy as np
from pathlib import Path
from glob import glob
from pydicom.filereader import dcmread
from pydicom.dataset import Dataset
from scipy import stats
from skimage.measure import (
    moments_hu,
    shannon_entropy,
    inertia_tensor_eigvals,
    blur_effect,
)

seg_sop = "1.2.840.10008.5.1.4.1.1.66.4"

dicom_header_dict = dict(
    study_uid=("0020", "000D"),
    series_uid=("0020", "000E"),
    diffusion_bvalue=("0018", "9087"),
    diffusion_directionality=("0018", "9075"),
    echo_time=("0018", "0081"),
    echo_train_length=("0018", "0091"),
    repetition_time=("0018", "0080"),
    flip_angle=("0018", "1314"),
    in_plane_phase_encoding_direction=("0018", "1312"),
    mr_acquisition_type=("0018", "0023"),
    acquisition_matrix=("0018", "1310"),
    patient_position=("0018", "5100"),
    reconstruction_matrix=("0018", "1100"),
    magnetic_field_strength=("0018", "0087"),
    manufacturer=("0008", "0070"),
    manufacturer_model_name=("0008", "1090"),
    number_of_phase_encoding_steps=("0018", "0089"),
    percent_phase_field_of_view=("0018", "0094"),
    pixel_bandwidth=("0018", "0095"),
    receive_coil_name=("0018", "1250"),
    transmit_coil_name=("0018", "1251"),
    sar=("0018", "1316"),
    scanning_sequence=("0018", "0020"),
    sequence_variant=("0018", "0021"),
    slice_thickness=("0018", "0050"),
    software_versions=("0018", "1020"),
    temporal_resolution=("0020", "0110"),
    image_orientation_patient=("0020", "0037"),
    image_type=("0008", "0008"),
    scan_options=("0018", "0022"),
    photometric_interpretation=("0028", "0004"),
    spectrally_selected_suppression=("0018", "9025"),
    inversion_time=("0018", "0082"),
    pixel_spacing=("0028", "0030"),
    number_of_echos=("0018", "0086"),
    number_of_temporal_positions=("0020", "0105"),
    modality=("0008", "0060"),
    series_description=("0008", "103E"),
    diffusion_bvalue_ge=("0043", "1039"),
    diffusion_bvalue_siemens=("0019", "100C"),
)

pcai_mapping = {
    "0010,0020": "patient_id",
    "0020,000d": "study_uid",
    "0020,000e": "series_uid",
    "0020,0013": "instance_number",
    "0018,9087": "diffusion_bvalue",
    "0018,9075": "diffusion_directionality",
    "0018,0081": "echo_time",
    "0018,0091": "echo_train_length",
    "0018,0080": "repetition_time",
    "0018,1314": "flip_angle",
    "0018,1312": "in_plane_phase_encoding_direction",
    "0018,0023": "mr_acquisition_type",
    "0018,1310": "acquisition_matrix",
    "0018,5100": "patient_position",
    "0018,1100": "reconstruction_matrix",  # actually reconstruction_diameter
    "0018,0087": "magnetic_field_strength",
    "0008,0070": "manufacturer",
    "0008,1090": "manufacturer_model_name",
    "0018,0089": "number_of_phase_encoding_steps",
    "0018,0094": "percent_phase_field_of_view",
    "0018,0095": "pixel_bandwidth",
    "0018,1250": "receive_coil_name",
    "0018,1251": "transmit_coil_name",
    "0018,1316": "sar",
    "0018,0020": "scanning_sequence",
    "0018,0021": "sequence_variant",
    "0018,0050": "slice_thickness",
    "0018,1020": "software_versions",
    "0020,0110": "temporal_resolution",
    "0020,0037": "image_orientation_patient",
    "0008,0008": "image_type",
    "0018,0022": "scan_options",
    "0028,0004": "photometric_interpretation",
    "0018,9025": "spectrally_selected_suppression",
    "0018,0082": "inversion_time",
    "0028,0030": "pixel_spacing",
    "0018,0086": "number_of_echos",
    "0020,0105": "number_of_temporal_positions",
    "0008,0060": "modality",
    "0008,103e": "series_description",
}


def extract_features_from_dicom(
    path: str, join: bool = True, return_paths: bool = False
) -> dict[str, list | str | float | int]:
    file_paths = glob(os.path.join(path, "*dcm"))
    n_images = len(file_paths)
    output_dict = {"number_of_images": n_images}
    for file in file_paths:
        try:
            dicom_file = dcmread(file, stop_before_pixels=True)
        except:
            continue

        for k in dicom_header_dict:
            dicom_key = dicom_header_dict[k]
            dicom_key = (
                eval("0x{}".format(dicom_key[0])),
                eval("0x{}".format(dicom_key[1])),
            )
            if k not in output_dict:
                output_dict[k] = []
            if dicom_key in dicom_file:
                v = dicom_file[dicom_key].value
                # replace times with empty space...
                if k == "series_description":
                    v = re.sub("[0-9]+/[0-9]+/[0-9]+", "", v)
                    v = re.sub("[0-9]+-[0-9]+-[0-9]+", "", v)
                    v = re.sub("[0-9]+:[0-9]+:[0-9]+", "", v)
                elif k == "diffusion_bvalue":
                    if isinstance(v, bytes):
                        v = int.from_bytes(v, byteorder="big")
                        if v > 5000:
                            v = dicom_file[dicom_key].value[0]
                elif k == "diffusion_bvalue_ge":
                    if isinstance(v, bytes):
                        v = v.decode().split("\\")[0]
                    elif isinstance(v, pydicom.multival.MultiValue):
                        v = v[0]
                    if len(str(v)) > 5:
                        v = str(v)[-4:].lstrip("0")
            else:
                v = "-"
            if isinstance(v, pydicom.multival.MultiValue):
                v = " ".join([str(x) for x in v])
            if isinstance(v, list):
                v = " ".join([str(x) for x in v])
            v = str(v)
            if v not in output_dict[k]:
                output_dict[k].append(v)

    if join == True:
        for k in output_dict:
            if k != "number_of_images":
                output_dict[k] = "|".join(output_dict[k])

    if return_paths == True:
        output_dict["file_paths"] = file_paths
        output_dict["path"] = path

    return output_dict


def extract_metadata_from_file(dicom_file: Dataset):
    if (0x0008, 0x0016) not in dicom_file:
        return None
    # skips file if SOP class is segmentation
    if dicom_file[0x0008, 0x0016].value == seg_sop:
        return None
    output_dict = {}
    for k in dicom_header_dict:
        dicom_key = dicom_header_dict[k]
        dicom_key = (
            eval("0x{}".format(dicom_key[0])),
            eval("0x{}".format(dicom_key[1])),
        )
        if dicom_key in dicom_file:
            v = dicom_file[dicom_key].value
            if k == "diffusion_bvalue_ge":
                v = eval(str(v))
                if isinstance(v, list) == False:
                    v = v.decode()
                    v = v.split("\\")
                    v = str(v[0])
                else:
                    v = str(v[0])
                if len(v) > 5:
                    v = v[-4:]
            # replace times with empty space...
            if k == "series_description":
                v = re.sub("[0-9]+/[0-9]+/[0-9]+", "", v)
                v = re.sub("[0-9]+-[0-9]+-[0-9]+", "", v)
                v = re.sub("[0-9]+:[0-9]+:[0-9]+", "", v)
        else:
            v = "-"
        if isinstance(v, pydicom.multival.MultiValue):
            v = " ".join([str(x) for x in v])
        if isinstance(v, list):
            v = " ".join([str(x) for x in v])
        v = str(v)
        output_dict[k] = v
    return output_dict


def extract_all_metadata_from_dicom(path, skip_seg=True):
    file_paths = glob(os.path.join(path, "*dcm"))
    n_images = len(file_paths)
    output_dict = {"number_of_images": n_images}
    is_seg = False
    is_valid = True
    for file in file_paths:
        dicom_file = dcmread(file, stop_before_pixels=True)
        # skips file if basic tag not present
        if (0x0008, 0x0016) not in dicom_file:
            is_valid = False
            continue
        # skips file if SOP class is segmentation
        if dicom_file[0x0008, 0x0016].value == seg_sop:
            is_seg = True
            continue
        for k in dicom_header_dict:
            dicom_key = dicom_header_dict[k]
            dicom_key = (
                eval("0x{}".format(dicom_key[0])),
                eval("0x{}".format(dicom_key[1])),
            )
            if k not in output_dict:
                output_dict[k] = []
            if dicom_key in dicom_file:
                v = dicom_file[dicom_key].value
                if k == "diffusion_bvalue_ge":
                    v = eval(str(v))
                    if isinstance(v, list) == False:
                        v = v.decode()
                        v = v.split("\\")
                        v = str(v[0])
                    else:
                        v = str(v[0])
                    if len(v) > 5:
                        v = v[-4:]
                # replace times with empty space...
                if k == "series_description":
                    v = re.sub("[0-9]+/[0-9]+/[0-9]+", "", v)
                    v = re.sub("[0-9]+-[0-9]+-[0-9]+", "", v)
                    v = re.sub("[0-9]+:[0-9]+:[0-9]+", "", v)
            else:
                v = "-"
            if isinstance(v, pydicom.multival.MultiValue):
                v = " ".join([str(x) for x in v])
            if isinstance(v, list):
                v = " ".join([str(x) for x in v])
            v = str(v)
            output_dict[k].append(v)

    output_dict["file_paths"] = file_paths
    output_dict["path"] = path
    output_dict["seg"] = is_seg
    output_dict["valid"] = is_valid

    return output_dict


def extract_pixel_features(pixel_array: Dataset) -> dict:
    """
    Extracts pixel-wise features from a pixel array.

    Args:
        pixel_array (Dataset): DICOM dataset.
    Returns:
        dict: pixel-wise features.
    """
    pixel_array = pixel_array.pixel_array.astype(np.float32)
    flat_array = pixel_array.flatten()
    features = {
        "mean": np.mean(pixel_array),
        "std": np.std(pixel_array),
        "min": np.min(pixel_array),
        "max": np.max(pixel_array),
        "median": np.median(pixel_array),
        "skew": stats.skew(flat_array),
        "kurtosis": stats.kurtosis(flat_array),
        "entropy": shannon_entropy(pixel_array),
        "rms": np.sqrt(np.mean(pixel_array**2)),
        "blur_effect": blur_effect(pixel_array),
        "x": pixel_array.shape[0],
        "y": pixel_array.shape[1],
    }
    moments = {f"moment_{i}": v for i, v in enumerate(moments_hu(pixel_array))}
    ev = {
        f"intertia_tensor_eigval_{i}": v
        for i, v in enumerate(inertia_tensor_eigvals(pixel_array))
    }
    features.update(moments)
    features.update(ev)
    features = {k: float(features[k]) for k in features}
    return features


def extract_pixel_features_series(path: str) -> dict[str, list]:
    """
    Extracts pixel-wise features from a series of DICOM files.
    Args:
        path (str): path to DICOM directory.
    Returns:
        dict: pixel-wise features.
    """
    file_paths = glob(os.path.join(path, "*dcm"))
    features = []
    all_keys = []
    for file in file_paths:
        dicom_file = dcmread(file)
        pixel_array = dicom_file.pixel_array
        features.append(extract_pixel_features(pixel_array))
        features[-1]["path"] = file
        for k in features[-1]:
            if k not in all_keys:
                all_keys.append(k)
    features = {k: [x[k] for x in features] for k in all_keys}
    return features


def extract_all_features(
    path: str, metadata_features: bool = True, pixel_features: bool = True
) -> dict:
    """
    Extracts all features from a DICOM file.

    Args:
        path (str): path to DICOM file.
        metadata_features (bool, optional): extract metadata features. Defaults
            to True.
        pixel_features (bool, optional): extract pixel features. Defaults to
            True.

    Returns:
        dict: all features.
    """

    study_uid, series_uid, file_name = str(path).split(os.sep)[-3:]
    features = {
        "study_uid": study_uid,
        "series_uid": series_uid,
        "file_name": file_name,
        "file_path": path,
    }
    dicom_file = dcmread(path)

    valid = True
    if metadata_features is True:
        try:
            features.update(extract_metadata_from_file(dicom_file))
        except:
            valid = False
    if pixel_features is True:
        try:
            features.update(extract_pixel_features(dicom_file))
        except:
            valid = False

    features["seg"] = dicom_file[0x0008, 0x0016].value == seg_sop
    features["valid"] = valid
    return features


def extract_all_features_series(
    path: str, metadata_features: bool = True, pixel_features: bool = True
) -> dict:
    """
    Extracts all features from a DICOM directory.

    Args:
        path (str): path to DICOM directory.
        metadata_features (bool, optional): extract metadata features. Defaults
            to True.
        pixel_features (bool, optional): extract pixel features. Defaults to
            True.

    Returns:
        dict: all features.
    """
    file_list = Path(path).rglob("*dcm")
    all_keys = []
    all_features = []
    for file in file_list:
        features = extract_all_features(file, metadata_features, pixel_features)
        for k in features:
            if k not in all_keys:
                all_keys.append(k)
        all_features.append(features)
    return {k: [x[k] for x in all_features] for k in all_keys}


if __name__ == "__main__":
    import sys
    import json

    print(json.dumps(extract_all_features_series(sys.argv[1]), indent=1))
