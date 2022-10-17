import os
import re
import pydicom
from glob import glob
from pydicom.filereader import read_file

dicom_header_dict = dict(
    study_uid=("0020","000D"),
    series_uid=("0020","000E"),
    diffusion_bvalue=("0018","9087"),
    diffusion_directionality=("0018","9075"),
    echo_time=("0018","0081"),
    echo_train_length=("0018","0091"),
    repetition_time=("0018","0080"),
    flip_angle=("0018","1314"),
    in_plane_phase_encoding_direction=("0018","1312"),
    mr_acquisition_type=("0018","0023"),
    acquisition_matrix=("0018","1310"),
    patient_position=("0018","5100"),
    reconstruction_matrix=("0018","1100"),
    magnetic_field_strength=("0018","0087"),
    manufacturer=("0008","0070"),
    manufacturer_model_name=("0008","1090"),
    number_of_phase_encoding_steps=("0018","0089"),
    percent_phase_field_of_view=("0018","0094"),
    pixel_bandwidth=("0018","0095"),
    receive_coil_name=("0018","1250"),
    transmit_coil_name=("0018","1251"),
    sar=("0018","1316"),
    scanning_sequence=("0018","0020"),
    sequence_variant=("0018","0021"),
    slice_thickness=("0018","0050"),
    software_versions=("0018","1020"),
    temporal_resolution=("0020","0110"),
    image_orientation_patient=("0020","0037"),
    image_type=("0008","0008"),
    scan_options=("0018","0022"),
    photometric_interpretation=("0028","0004"),
    spectrally_selected_suppression=("0018","9025"),
    inversion_time=("0018","0082"),
    pixel_spacing=("0028","0030"),
    number_of_echos=("0018","0086"),
    number_of_temporal_positions=("0020","0105"),
    modality=("0008","0060"),
    series_description=("0008","103E")
)

def extract_features_from_dicom(path,join=True,return_paths=False):
    file_paths = glob(os.path.join(path,"*dcm"))
    n_images = len(file_paths)
    output_dict = {"number_of_images":n_images}
    for file in file_paths:
        dicom_file = read_file(file)

        for k in dicom_header_dict:
            dicom_key = dicom_header_dict[k]
            dicom_key = (eval("0x{}".format(dicom_key[0])),
                        eval("0x{}".format(dicom_key[1])))
            if k not in output_dict:
                output_dict[k] = []
            if dicom_key in dicom_file:
                v = dicom_file[dicom_key].value
                # replace times with empty space...
                if k == "series_description":
                    v = re.sub("[0-9]+/[0-9]+/[0-9]+", "", v)
                    v = re.sub("[0-9]+-[0-9]+-[0-9]+", "", v)
                    v = re.sub("[0-9]+:[0-9]+:[0-9]+", "", v)
            else:
                v = "-"
            if isinstance(v,pydicom.multival.MultiValue):
                v = " ".join([str(x) for x in v])
            if isinstance(v,list):
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

def extract_all_metadata_from_dicom(path):
    file_paths = glob(os.path.join(path,"*dcm"))
    n_images = len(file_paths)
    output_dict = {"number_of_images":n_images}
    for file in file_paths:
        dicom_file = read_file(file)
        for k in dicom_header_dict:
            dicom_key = dicom_header_dict[k]
            dicom_key = (eval("0x{}".format(dicom_key[0])),
                        eval("0x{}".format(dicom_key[1])))
            if k not in output_dict:
                output_dict[k] = []
            if dicom_key in dicom_file:
                v = dicom_file[dicom_key].value
                # replace times with empty space...
                if k == "series_description":
                    v = re.sub("[0-9]+/[0-9]+/[0-9]+", "", v)
                    v = re.sub("[0-9]+-[0-9]+-[0-9]+", "", v)
                    v = re.sub("[0-9]+:[0-9]+:[0-9]+", "", v)
            else:
                v = "-"
            if isinstance(v,pydicom.multival.MultiValue):
                v = " ".join([str(x) for x in v])
            if isinstance(v,list):
                v = " ".join([str(x) for x in v])
            v = str(v)
            output_dict[k].append(v)
        
    output_dict["file_paths"] = file_paths
    output_dict["path"] = path

    return output_dict

if __name__ == "__main__":
    import sys

    print(extract_features_from_dicom(sys.argv[1]))