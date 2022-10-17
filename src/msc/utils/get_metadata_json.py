import argparse
import os
import json
import re
import numpy as np
from multiprocessing import Pool
from glob import glob
from tqdm import tqdm

from ..dicom_feature_extraction import extract_all_metadata_from_dicom

def filter_b_value(d:dict)->dict:
    """Filters the metadata dictionary and keeps only entries with the
    maximum b-value.

    Args:
        d (dict): metadata dictionary.

    Returns:
        dict: metadata dictionary with only the highest b-value entries.
    """
    try:
        bval = np.array(d["diffusion_bvalue"])
        bval[bval == "-"] = "0.0"
        bval = np.float32(bval)
        s = np.unique(bval)
        if len(s) > 1:
            max_bval = np.max(s)
            for k in d:
                if k not in ["number_of_images","path"]:
                    d[k] = [x for i,x in enumerate(d[k]) if bval[i] == max_bval]
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
    d = filter_b_value(d)
    return d

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
            metadata = wraper(f)
            sd = metadata["series_description"]
            if individual_id not in all_metadata:
                all_metadata[individual_id] = []
            all_metadata[individual_id].append(metadata)
    
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
                    if individual_id not in all_metadata:
                        all_metadata[individual_id] = []
                    all_metadata[individual_id].append(o)
                prog.update()
                batch = []

        if len(batch) > 0:
            output = p.map(wraper,batch)
            for o in output:
                individual_id = re.search(
                    args.individual_pattern,o["path"]).group()
                if individual_id not in all_metadata:
                    all_metadata[individual_id] = []
                all_metadata[individual_id].append(o)
            batch = []
            prog.update()
        prog.close()

    print(json.dumps(all_metadata,indent=2,sort_keys=True))