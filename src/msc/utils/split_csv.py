import argparse
import numpy as np
import pandas as pd
from numpy.random import default_rng
from pandas.api.types import is_string_dtype

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Splits a CSV-type file into two different sets.")

    parser.add_argument("--input_path",dest="input_path",
                        type=str,required=True)
    parser.add_argument("--sep",dest="sep",default=None,
                        type=str,required=False)
    parser.add_argument("--id_col",dest="id_col",default=None,
                        type=str)
    parser.add_argument("--output_paths",dest="output_paths",nargs=2,
                        type=str,required=True)
    parser.add_argument("--split_ratio",dest="split_ratio",
                        type=float,default=0.8)
    parser.add_argument("--random_seed",dest="random_seed",default=42,
                        type=int)
    
    args = parser.parse_args()

    df = pd.read_csv(args.input_path,sep=args.sep,engine="python")
    for col in df.columns:
        if is_string_dtype(df[col]):
            df[col] = df[col].str.replace('"', '')

    rng = default_rng(seed=args.random_seed)

    if args.id_col is None:
        split = rng.choice(
            df.shape[0],int(args.split_ratio*df.shape[0]),replace=False)
        split_a = split
        split_b = [i for i in range(df.shape[0]) if i not in split_a]
    else:
        ids = df[args.id_col]
        unique_ids = np.unique(ids)
        id_split = rng.choice(
            unique_ids,int(args.split_ratio*len(unique_ids)),replace=False)
        id_split_a = id_split
        id_split_b = [i for i in unique_ids 
                      if i not in id_split]
        split_a = [i for i,x in enumerate(ids) if x in id_split_a]
        split_b = [i for i,x in enumerate(ids) if x in id_split_b]

    df_a = df.loc[split_a]
    df_b = df.loc[split_b]

    df_a.to_csv(args.output_paths[0],index=False)
    df_b.to_csv(args.output_paths[1],index=False)