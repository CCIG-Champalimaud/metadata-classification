import argparse
import numpy as np
import pandas as pd

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Cross validates models.")

    parser.add_argument("--input_path",dest="input_path",
                        type=str,required=True)
    parser.add_argument("--test_set_path",dest="test_set_path",
                        type=str,default=None)
    parser.add_argument("--output_path",dest="output_path",
                        type=str,required=True)
    parser.add_argument("--n_folds",dest="n_folds",
                        type=int,default=10)
    args = parser.parse_args()

    # load data, fix possible issues with it 
    test_data_path = "data/data_train.csv"
    data_hold_out = pd.read_csv(test_data_path)
    data_hold_out.loc[data_hold_out["class"] == "DCE","class"] = "dce"

    data_hold_out = sanitize_data(data_hold_out)
    data_hold_out = sequence_to_other_df(
        data_hold_out,"study_uid","class","dwi","adc","percent_phase_field_of_view")
    data_hold_out = sequence_to_other_df(
        data_hold_out,"study_uid","class","dwi","adc","sar")

    X_hold_out = data_hold_out.drop(cols_to_drop,axis=1)
    y_hold_out = np.array(data_hold_out["class"])