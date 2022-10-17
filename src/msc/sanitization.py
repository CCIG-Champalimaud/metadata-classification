import numpy as np
import pandas as pd
from .constants import (
    text_sep_cols,num_sep_cols,num_cols,replace_cols,cols_to_drop)

def sanitize_input(X,replacements={},strips=[]):
    for k in replacements:
        new_X = []
        for x in X:
            if isinstance(x,str) == True:
                x = x.replace(k,replacements[k])
            new_X.append(x)
        X = new_X
    for s in strips:
        new_X = []
        for x in X:
            if isinstance(x,str) == True:
                x = x.strip(s)
            new_X.append(x)
        X = new_X
    for i in range(len(X)):
        if isinstance(X[i],str) == True:
            X[i] = X[i].lower()
    return np.array(X)

rep_dict = {"|":" ","-":" ",
            ";":" ",",":" ",".":" ",
            "(":" ",")":" ",
            "_":" ",":":" "}

def sanitize_data(data):
    for k in text_sep_cols:
        data[k] = sanitize_input(
            data[k],rep_dict,[" "])
    for k in num_sep_cols:
        data[k] = sanitize_input(
            data[k],rep_dict,[" "])
    for k in replace_cols:
        data.loc[data[k]=="",k] = replace_cols[k]
    return data

def sequence_to_other(X,key_col,from_key,to_key,val_col):
    if any(from_key == X[key_col]) and any(to_key == X[key_col]):
        value = X[val_col][X[key_col] == from_key].tolist()[0]
        X[val_col][X[key_col] == to_key] = value
    return X

def sequence_to_other_df(X,group,key_col,from_key,to_key,val_col):
    return X.groupby(group).apply(
        lambda x: sequence_to_other(
            x,key_col,from_key,to_key,val_col))

def data_loading_wraper(data_path,infer=False):
    # load data, fix some minor recurring issues
    data = pd.read_csv(data_path)
    data.loc[data["class"] == "DCE","class"] = "dce"

    # sanitize data
    data = sanitize_data(data)
    if infer == True:
        data = sequence_to_other_df(
            data,"study_uid","class","dwi","adc","percent_phase_field_of_view")
        data = sequence_to_other_df(
            data,"study_uid","class","dwi","adc","sar")

    X = data.drop(cols_to_drop,axis=1)
    y = np.array(data["class"])
    study_uids = data["study_uid"]
    unique_study_uids = list(set(study_uids))
    return X,y,study_uids,unique_study_uids