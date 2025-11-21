import argparse
import os
import json
import re
import logging
import numpy as np
from multiprocessing import Pool
from pathlib import Path
from tqdm import tqdm

from ..dicom_feature_extraction import extract_features_from_dicom


logger = logging.getLogger(__name__)


def filter_b_value(d: dict, bval_key: str = "diffusion_bvalue") -> dict:
    """
    Filters the metadata dictionary and keeps only entries with the maximum
    b-value.

    Args:
        d (dict): metadata dictionary.
        bval_key (str, optional): key corresponding to b-value

    Returns:
        dict: metadata dictionary with only the highest b-value entries.
    """
    try:
        bval = np.array(d[bval_key])
        bval[bval == "-"] = "-1"
        bval = np.float32(bval)
        s = np.unique(bval)
        if len(s) > 1:
            max_bval = np.max(s)
            logger.debug(
                "Filtering by maximum b-value",
                extra={"bval_key": bval_key, "max_bval": float(max_bval)},
            )
            for k in d:
                if k not in ["number_of_images", "path"]:
                    d[k] = [
                        x for i, x in enumerate(d[k]) if bval[i] == max_bval
                    ]
    except Exception as e:
        logger.warning(
            "Failed to filter by b-value",
            extra={"bval_key": bval_key, "error": str(e)},
        )
    return d


def wraper(p: str) -> dict:
    """Wraps the metadata extraction and b-value filtering from a DICOM
    directory path.

    Args:
        p (str): path to DICOM directory.

    Returns:
        d (dict): metadata dictionary.
    """
    d = extract_features_from_dicom(p, join=False, return_paths=True)
    if len(d["file_paths"]) > 0 and d["seg"] == False and d["valid"] == True:
        d = filter_b_value(d)
        # siemens and ge medical systems store the b-values differently by default
        if "siemens" in d["manufacturer"][0].lower():
            d = filter_b_value(d, "diffusion_bvalue_siemens")
        if "ge med" in d["manufacturer"][0].lower():
            d = filter_b_value(d, "diffusion_bvalue_ge")
    return d


def update_dict(
    dictionary: dict,
    individual_id: str,
    study_id: str,
    sequence_id: str,
    d: dict,
) -> dict:
    """
    Updates a hierarchical dictionary with entries per individual, study and
    sequence unique identifiers (four levels of depth including features).

    Args:
        dictionary (dict): dictionary to be updated.
        individual_id (str): individual ID (first level).
        study_id (str): study ID (second level).
        sequence_id (str): sequence ID (third level).
        d (dict): feature dictionary.

    Returns:
        dict: updated version of dictionary.
    """
    if len(d["file_paths"]) > 0 and d["seg"] == False:
        if individual_id not in dictionary:
            dictionary[individual_id] = {}
        if study_id not in dictionary[individual_id]:
            dictionary[individual_id][study_id] = {}

        dictionary[individual_id][study_id][sequence_id] = d
    return dictionary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "Extracts the metadata of a directory containing DICOM series and \
            prints the data in JSON format."
    )

    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--pattern", default="*")
    parser.add_argument("--individual_pattern", default="*")
    parser.add_argument("--n_workers", default=0, type=int)

    args = parser.parse_args()

    logger.info(
        "Starting metadata extraction to JSON",
        extra={
            "input_dir": args.input_dir,
            "pattern": args.pattern,
            "individual_pattern": args.individual_pattern,
            "n_workers": args.n_workers,
        },
    )

    all_metadata = {}
    files = [str(x) for x in Path(args.input_dir).glob(args.pattern)]
    logger.info("Found %d files for metadata extraction", len(files))
    if args.n_workers < 2:
        for f in tqdm(files):
            individual_id = re.search(args.individual_pattern, f).group()
            study_id, sequence_id = f.split(os.sep)[-2:]
            metadata = wraper(f)
            all_metadata = update_dict(
                all_metadata, individual_id, study_id, sequence_id, metadata
            )
    else:
        p = Pool(args.n_workers)
        batch = []
        l = args.n_workers * 8
        prog = tqdm(total=len(files) // l + (len(files) % l > 0))
        for f in files:
            batch.append(f)
            if len(batch) > l:
                output = p.map(wraper, batch)
                for o in output:
                    individual_id = re.search(
                        args.individual_pattern, o["path"]
                    ).group()
                    study_id, sequence_id = f.split(os.sep)[-2:]
                    all_metadata = update_dict(
                        all_metadata, individual_id, study_id, sequence_id, o
                    )
                prog.update()
                batch = []

        if len(batch) > 0:
            output = p.map(wraper, batch)
            for o in output:
                individual_id = re.search(
                    args.individual_pattern, o["path"]
                ).group()
                study_id, sequence_id = f.split(os.sep)[-2:]
                all_metadata = update_dict(
                    all_metadata, individual_id, study_id, sequence_id, o
                )
            batch = []
            prog.update()
        prog.close()

    logger.info(
        "Finished metadata extraction", extra={"n_patients": len(all_metadata)}
    )
    print(json.dumps(all_metadata, indent=2, sort_keys=True))
