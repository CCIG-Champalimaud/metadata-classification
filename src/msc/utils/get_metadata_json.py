import argparse
import os
import json
import re
import numpy as np
from multiprocessing import Pool
from glob import glob
from tqdm import tqdm

from ..dicom_feature_extraction import extract_all_metadata_from_dicom

def filter_b_value(d:dict,bval_key:str="diffusion_bvalue")->dict:
    """Filters the metadata dictionary and keeps only entries with the
    maximum b-value.

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
            for k in d:
                if k not in ["number_of_images","path"]:
                    d[k] = [x for i,x in enumerate(d[k]) 
                            if bval[i] == max_bval]
    except:
        pass
    return d

def wraper(p:str)->dict:
    """Wraps the metadata extraction and b-value filtering from a DICOM 
    directory path.

    Args:
        p (str): path to DICOM directory.

    Returns:
        d (dict): metadata dictionary.
    """
    d = extract_all_metadata_from_dicom(p)
    if len(d["file_paths"]) > 0 and d["seg"] == False:
        d = filter_b_value(d)
        # siemens and ge medical systems store the b-values differently by default
        if "siemens" in d["manufacturer"][0].lower():
            d = filter_b_value(d,"diffusion_bvalue_siemens")
        if "ge med" in d["manufacturer"][0].lower():
            d = filter_b_value(d,"diffusion_bvalue_ge")
    return d

def update_dict(dictionary,individual_id,study_id,sequence_id,d):
    if len(metadata["file_paths"]) > 0 and metadata["seg"] == False:
        if individual_id not in dictionary:
            dictionary[individual_id] = {}
        if study_id not in dictionary[individual_id]:
            dictionary[individual_id][study_id] = {}
        
        dictionary[individual_id][study_id][sequence_id] = metadata
    return dictionary

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "Extracts the metadata of a directory containing DICOM series and \
            prints the data in JSON format.")
    
    parser.add_argument("--input_dir",required=True)
    parser.add_argument("--pattern",default="*")
    parser.add_argument("--individual_pattern",default="*")
    parser.add_argument("--n_workers",default=0,type=int)
    
    args = parser.parse_args()
    
    all_metadata = {}
    files = glob(os.path.join(args.input_dir,args.pattern))
    if args.n_workers < 2:
        for f in tqdm(files):
            individual_id = re.search(args.individual_pattern,f).group()
            study_id,sequence_id = f.split(os.sep)[-2:]
            metadata = wraper(f)
            all_metadata = update_dict(
                all_metadata,individual_id,study_id,sequence_id,metadata)
    else:
        p = Pool(args.n_workers)
        batch = []
        l = args.n_workers * 8
        print("Processing data in batches with {} files".format(l))
        prog = tqdm(total=len(files) // l + (len(files) % l > 0))
        for f in files:
            batch.append(f)
            if len(batch) > l:
                output = p.map(wraper,batch)
                for o in output:
                    individual_id = re.search(
                        args.individual_pattern,o["path"]).group()
                    study_id,sequence_id = f.split(os.sep)[-2:]
                    all_metadata = update_dict(
                        all_metadata,individual_id,study_id,sequence_id,o)
                prog.update()
                batch = []

        if len(batch) > 0:
            output = p.map(wraper,batch)
            for o in output:
                individual_id = re.search(
                    args.individual_pattern,o["path"]).group()
                study_id,sequence_id = f.split(os.sep)[-2:]
                all_metadata = update_dict(
                    all_metadata,individual_id,study_id,sequence_id,o)
            batch = []
            prog.update()
        prog.close()

    print(json.dumps(all_metadata,indent=2,sort_keys=True))