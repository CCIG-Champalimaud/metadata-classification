import argparse
import numpy as np
import pandas as pd

from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler
from skopt import BayesSearchCV

from .constants import *
from .feature_extraction import *
from .sanitization import *
from .train_utils import *

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
    parser.add_argument("--model_name",dest="model_name",
                        default="rf",choices=["rf","elastic","extra_trees"])
    parser.add_argument("--bayesian_hp_opt",dest="bayesian_hp_opt",
                        action="store_true")
    args = parser.parse_args()

    X,y,study_uids,unique_study_uids = data_loading_wraper(args.input_path)

    kf = KFold(args.n_folds,shuffle=True,random_state=42)

    nan_remover = RemoveNan()

    fold_results = {"cv":[]}

    f = 0

    model_name = args.model_name
    bayesian = args.bayesian

    for train_idxs,val_idxs in kf.split(unique_study_uids):
        print("Fold {}".format(f+1))
        # data splitting
        train_uids = [unique_study_uids[i] for i in train_idxs]
        val_uids = [unique_study_uids[i] for i in val_idxs]
        train_idxs_long = [i for i,x in enumerate(study_uids) 
                        if x in train_uids]
        val_idxs_long = [i for i,x in enumerate(study_uids) 
                        if x in val_uids]
        training_X = X.iloc[train_idxs_long]
        training_y = y[train_idxs_long]
        val_X = X.iloc[val_idxs_long]
        val_y = y[val_idxs_long]

        print("\tTransforming data")
        count_vec = TextColsToCounts(
            text_cols={i:x for i,x in enumerate(X.columns) if x in text_sep_cols},
            text_num_cols={i:x for i,x in enumerate(X.columns) if x in num_sep_cols},
            num_cols={i:x for i,x in enumerate(X.columns) if x in num_cols})
        count_vec.fit(training_X)
        out = count_vec.transform(training_X)
        out,training_y = nan_remover.transform(out,training_y)
        out_val = count_vec.transform(val_X)
        out_val,val_y = nan_remover.transform(out_val,val_y)

        print("\tTraining model")
        if bayesian == True:
            md = model_dict_bayes
        else:
            md = model_dict

        p = Pipeline(
            [("remove_zero_var",VarianceThreshold()),
            ("standardise",StandardScaler()),
            ("model",md[model_name]["model"](**md[model_name]["params"]))]
        )

        if bayesian == True:
            search_fn = BayesSearchCV
        else:
            search_fn = GridSearchCV

        model = search_fn(
            p,
            {"model__"+k:md[model_name]["cv_params"][k]
                for k in md[model_name]["cv_params"]},
            verbose=0,n_jobs=4,cv=3,scoring="f1_macro")

        model.fit(out,training_y)

        print("\tEvaluating model")
        y_pred = model.predict(out_val)
        fold_results["cv"].append({
            "auc":roc_auc_score(val_y,model.predict_proba(out_val),
                                multi_class="ovr"),
            "count_vec":count_vec,
            "model":model,
            "y_true":val_y,
            "y_pred":y_pred,
            })
        
        f += 1

    if args.test_set_path is not None:
        X_ho,y_ho,_,_ = data_loading_wraper(args.test_set_path)
        fold_results["test"] = []
        for i,f in enumerate(fold_results["cv"]):
            count_vec = f["count_vec"]
            model = f["model"]
            test_data_pred = count_vec.transform(X_ho)
            test_data_pred,y_test = nan_remover.transform(test_data_pred,y_ho)

            y_pred = model.predict(test_data_pred)

            fold_results["test"].append(
                {"y_true":y_test,"y_pred":y_pred,
                 "auc":roc_auc_score(y_test,model.predict_proba(test_data_pred),
                                     multi_class="ovr")})
