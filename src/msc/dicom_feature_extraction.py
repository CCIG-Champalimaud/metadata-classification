import os
import re
import pydicom
import numpy as np
from typing import Any
from pathlib import Path
from glob import glob
from pydicom.filereader import dcmread
from pydicom.dataset import Dataset
from scipy import stats

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

inverted_dicom_header_dict = {v: k for k, v in dicom_header_dict.items()}

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

image_feature_keys = [
    "image_mean",
    "image_std",
    "image_min",
    "image_max",
    "image_median",
    "image_skew",
    "image_kurtosis",
    "image_entropy",
    "image_rms",
    "image_blur_effect",
    "image_x",
    "image_y",
    "image_moment_0",
    "image_moment_1",
    "image_moment_2",
    "image_moment_3",
    "image_moment_4",
    "image_moment_5",
    "image_moment_6",
    "image_inertia_tensor_eigval_0",
    "image_inertia_tensor_eigval_1",
    "image_inertia_tensor_eigval_2",
    "image_inertia_tensor_eigval_3",
]

converted_dicom_header_dict = {
    k: (eval("0x{}".format(v[0])), eval("0x{}".format(v[1])))
    for k, v in dicom_header_dict.items()
}


def process_bvalue(v: bytes | float | int, dicom_file: Dataset) -> float | int:
    """
    Process b-value from DICOM header.

    Args:
        v (bytes | float | int): b-value.
        dicom_file (Dataset): DICOM file.

    Returns:
        float | int: processed b-value.
    """
    if isinstance(v, bytes):
        v = int.from_bytes(v, byteorder="big")
        if v > 5000:
            v = dicom_file[dicom_header_dict["diffusion_bvalue"]].value[0]
    return v


def process_bvalue_ge(
    v: bytes | float | int, dicom_file: None = None
) -> float | int:
    """
    Process GE scanner b-value from DICOM header.

    Args:
        v (bytes | float | int): b-value.
        dicom_file (None): only for compatibility purposes.

    Returns:
        float | int: processed b-value.
    """
    if isinstance(v, bytes):
        v = v.decode().split("\\")[0]
    elif isinstance(v, pydicom.multival.MultiValue):
        v = v[0]
    if len(str(v)) > 5:
        v = str(v)[-4:].lstrip("0")
    return v


def process_series_description(v: str, dicom_file: None = None) -> str:
    """
    Process series description from DICOM header.

    Args:
        v (str): series description.
        dicom_file (None): only for compatibility purposes.

    Returns:
        str: processed series description.
    """
    # replace times with empty space...
    v = re.sub("[0-9]+/[0-9]+/[0-9]+", "", v)
    v = re.sub("[0-9]+-[0-9]+-[0-9]+", "", v)
    v = re.sub("[0-9]+:[0-9]+:[0-9]+", "", v)
    return v


def process_multivalues(v: Any) -> str | int | float:
    """
    Process multi-valued DICOM header.

    Args:
        v (Any): possibly multi-valued DICOM header.

    Returns:
        (str | int | float): processed multi-valued DICOM header.
    """
    if isinstance(v, pydicom.multival.MultiValue):
        v = " ".join([str(x) for x in v])
    if isinstance(v, list):
        v = " ".join([str(x) for x in v])
    v = str(v)
    return v


value_processors = {
    "diffusion_bvalue": process_bvalue,
    "diffusion_bvalue_ge": process_bvalue_ge,
    "series_description": process_series_description,
}


def extract_metadata_from_file(dicom_file: Dataset) -> dict:
    """
    Extract metadata from DICOM file.

    Args:
        dicom_file (Dataset): opened dicom file.

    Returns:
        dict: extracted metadata (available in ``dicom_header_dict``).
    """
    output_dict = {"valid": True, "seg": False}
    if (0x0008, 0x0016) not in dicom_file:
        dicom_file["valid"] = False
    # skips file if SOP class is segmentation
    if dicom_file[0x0008, 0x0016].value == seg_sop:
        dicom_file["seg"] = True
    for k in dicom_header_dict:
        dicom_key = converted_dicom_header_dict[k]
        if dicom_key in dicom_file:
            v = dicom_file[dicom_key].value
            if k in value_processors:
                v = value_processors[k](v, dicom_file)
        else:
            v = "-"
        v = process_multivalues(v)
        output_dict[k] = v

    output_dict["diffusion_bvalue_final"] = "-"
    for bvalue_key in [
        "diffusion_bvalue",
        "diffusion_bvalue_ge",
        "diffusion_bvalue_siemens",
    ]:
        if output_dict[bvalue_key] != "-":
            output_dict["diffusion_bvalue_final"] = output_dict[bvalue_key]
            break

    return output_dict


def extract_features_from_dicom(
    path: str,
    join: bool = False,
    return_paths: bool = False,
    image_features: bool = True,
) -> dict[str, list | str | float | int]:
    """
    Extract features (specified in ``dicom_header_dict``) from DICOM files in a
    given folder.

    Args:
        path (str): path to folder containing DICOM files.
        join (bool, optional): whether the unique outputs should be joined using
            ``'|'`` after feature extraction. Defaults to True.
        return_paths (bool, optional): whether paths should be returned.
            Defaults to False.
        image_features (bool, optional): whether image features should be
            extracted. Defaults to False.

    Returns:
        dict[str, list | str | float | int]: a dictionary with features.
    """
    file_paths = glob(os.path.join(path, "*dcm"))
    n_images = len(file_paths)
    output_dict = {}
    N = 0
    for file in file_paths:
        try:
            dicom_file = dcmread(file, stop_before_pixels=not image_features)
        except:
            continue

        features = extract_metadata_from_file(dicom_file)
        if features is None:
            continue
        if image_features is True and (0x7FE0, 0x0010) in dicom_file:
            try:
                features.update(extract_pixel_features(dicom_file))
            except ValueError:
                # can happen if incomplete pixel array is present
                continue

        for k in features:
            if k not in output_dict:
                output_dict[k] = []
            output_dict[k].append(features[k])
        N += 1

    output_dict["number_of_images"] = [n_images for _ in range(N)]
    if join is True:
        for k in output_dict:
            if (
                k not in ["number_of_images", "seg", "valid"]
                and k not in image_feature_keys
            ):
                output_dict[k] = "|".join(set(list(output_dict[k])))

    if return_paths == True:
        output_dict["file_paths"] = file_paths
        output_dict["path"] = path

    return output_dict


def extract_pixel_features(dicom_file: Dataset) -> dict:
    """
    Extracts pixel-wise features from a pixel array.

    Args:
        dicom_file (Dataset): DICOM dataset.
    Returns:
        dict: pixel-wise features.
    """
    try:
        from skimage.measure import (
            moments_hu,
            shannon_entropy,
            inertia_tensor_eigvals,
            blur_effect,
        )
    except ImportError:
        raise ImportError(
            "The scikit-image is required to extract pixel-wise features."
        )
    pixel_array = dicom_file.pixel_array.astype(np.float32)
    flat_array = pixel_array.flatten()
    features = {
        "image_mean": np.mean(pixel_array),
        "image_std": np.std(pixel_array),
        "image_min": np.min(pixel_array),
        "image_max": np.max(pixel_array),
        "image_median": np.median(pixel_array),
        "image_skew": stats.skew(flat_array),
        "image_kurtosis": stats.kurtosis(flat_array),
        "image_entropy": shannon_entropy(pixel_array),
        "image_rms": np.sqrt(np.mean(pixel_array**2)),
        "image_blur_effect": blur_effect(pixel_array),
        "image_x": pixel_array.shape[0],
        "image_y": pixel_array.shape[1],
    }
    moments = {
        f"image_moment_{i}": v for i, v in enumerate(moments_hu(pixel_array))
    }
    ev = {
        f"image_inertia_tensor_eigval_{i}": v
        for i, v in enumerate(inertia_tensor_eigvals(pixel_array))
    }
    features.update(moments)
    features.update(ev)
    features = {k: float(features[k]) for k in features}
    return features


def extract_pixel_features_series(path: str) -> dict[str, list]:
    """
    Extracts image features from a series of DICOM files using
    ``extract_pixel_features``.

    Args:
        path (str): path to DICOM directory.

    Returns:
        dict: image features.
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
    path: str, metadata_features: bool = True, image_features: bool = True
) -> dict:
    """
    Extracts all features from a DICOM file.

    Args:
        path (str): path to DICOM file.
        metadata_features (bool, optional): extract metadata features. Defaults
            to True.
        image_features (bool, optional): extract image features. Defaults to
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
    valid = True
    try:
        dicom_file = dcmread(path)
    except:
        valid = False

    if valid:
        if metadata_features is True:
            try:
                features.update(extract_metadata_from_file(dicom_file))
            except:
                valid = False
        if image_features is True:
            try:
                features.update(extract_pixel_features(dicom_file))
            except:
                valid = False
        features["seg"] = dicom_file[0x0008, 0x0016].value == seg_sop
    else:
        features["seg"] = False
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
