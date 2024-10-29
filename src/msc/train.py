import argparse
import re
import numpy as np
import dill
import json
import fasttreeshap as shap
from pathlib import Path
from sklearn.metrics import roc_auc_score, r2_score
from sklearn.pipeline import Pipeline
from sklearn.model_selection import (
    RandomizedSearchCV,
    KFold,
    StratifiedKFold,
    LeaveOneGroupOut,
)
from sklearn.feature_selection import VarianceThreshold
from catboost import EFstrType, Pool, FeaturesData

from .feature_extraction import RemoveNan, TextColsToCounts
from .sanitization import text_sep_cols, num_sep_cols, num_cols
from .data_loading import data_loading_wraper
from .train_utils import model_dict

N_ITER = 50


def read_params(expr_list: list[str]) -> dict:
    """
    Reads a list of expressions defined as `key=value` and returns a dict after
    cohercing to python types.

    Args:
        expr_list (list[str]): list of expressions defined as `key=value`

    Returns:
        dict: dict of key-value pairs.
    """

    def get_type(val: str):
        val = val.strip()
        if re.match("^[-]*[0-9]+$", val):
            val = int(val)
        elif re.match("^[-]*[0-9\.]+$", val):
            val = float(val)
        elif val == True:
            val = True
        elif val == False:
            val = True
        # list, tuples, sets
        elif re.match("^[\[|\(\{]{1}[0-9A-Za-z ,]+[\]|\)\}]{1}$", val):
            convert_fn = {"[": list, "{": set, "(": tuple}[val[0]]
            val = val.strip("[]").strip("()").split(",")
            val = convert_fn([get_type(v) for v in val])
        elif re.match("\{[0-9A-Za-z ,:]+\}", val):
            val = val.strip("{}").split(",")
            val = [v.split(":") for v in val]
            val = {get_type(k): get_type(v) for k, v in val}
        return val

    params = {}
    for expr in expr_list:
        key, val = expr.split("=")
        params[key] = get_type(val)

    return params


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cross validates models.")

    parser.add_argument(
        "--input_path", type=str, required=True, help="Path to input CSV file"
    )
    parser.add_argument(
        "--test_set_path",
        type=str,
        default=None,
        help="Path to input test CSV file",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Output path for model file",
    )
    parser.add_argument(
        "--target_column",
        default="class",
        type=str,
        help="Name of the column containing the target",
    )
    parser.add_argument(
        "--n_folds",
        type=int,
        default=10,
        help="Number of folds (if --folds is not specified)",
    )
    parser.add_argument(
        "--folds", type=str, default=None, help="File with folds"
    )
    parser.add_argument(
        "--n_jobs", type=int, default=0, help="Number of parallel jobs"
    )
    parser.add_argument(
        "--model_name",
        default="rf",
        choices=["rf", "elastic", "extra_trees", "lgb", "xgb", "catboost"],
        help="Name of model",
    )
    parser.add_argument(
        "--exclude_cols",
        nargs="+",
        default=None,
        help="Columns to exclude from training and inference",
    )
    parser.add_argument(
        "--include_cols",
        nargs="+",
        default=None,
        help="Columns to include (exclusively) in training and inference",
    )
    parser.add_argument(
        "--save_model",
        action="store_true",
        help="Whether the model should be saved (if True only the metrics are \
            saved)",
    )
    parser.add_argument(
        "--random_seed",
        dest="random_seed",
        default=42,
        type=int,
        help="Random seed",
    )
    parser.add_argument(
        "--subset",
        dest="subset",
        default=None,
        type=float,
        help="Subsets data. Helpful for debugging or learning curves",
    )
    parser.add_argument(
        "--skip_feature_importance",
        default=False,
        action="store_true",
        help="Skips feature importance",
    )
    parser.add_argument(
        "--task_name",
        default="classification",
        choices=["classification", "regression"],
        help="Task type",
    )
    parser.add_argument(
        "--model_params",
        default=None,
        type=str,
        nargs="+",
        help="List of model parameters as key=value pairs (values are coherced to python values)",
    )
    args = parser.parse_args()

    RNG = np.random.default_rng(args.random_seed)

    if args.model_params is not None:
        params_override = read_params(args.model_params)
    else:
        params_override = {}

    X, y, study_uids, unique_study_uids = data_loading_wraper(
        args.input_path, target_column=args.target_column, task=args.task_name
    )

    if args.subset is not None:
        if args.subset > 1.0:
            subset = int(args.subset)
        else:
            subset = int(args.subset * X.shape[0])
        idxs = RNG.choice(X.shape[0], subset, replace=False)
        X = X[idxs]
        y = y[idxs]
        study_uids = study_uids[idxs]
    if args.folds is None:
        kf = KFold(args.n_folds, shuffle=True, random_state=args.random_seed)
        groups = None
    else:
        with open(args.folds) as o:
            groups_dict = json.load(o)

        unique_study_uids = [
            uid for uid in unique_study_uids if uid in groups_dict
        ]
        groups = [groups_dict[k] for k in unique_study_uids]
        kf = LeaveOneGroupOut()

    nan_remover = RemoveNan()
    train_results = {"cv": [], "args": vars(args)}
    f = 0
    model_name = args.model_name

    if args.exclude_cols is not None:
        text_sep_cols = [x for x in text_sep_cols if x not in args.exclude_cols]
        num_sep_cols = [x for x in num_sep_cols if x not in args.exclude_cols]
        num_cols = [x for x in num_cols if x not in args.exclude_cols]

    if args.include_cols is not None:
        text_sep_cols = [x for x in text_sep_cols if x in args.include_cols]
        num_sep_cols = [x for x in num_sep_cols if x in args.include_cols]
        num_cols = [x for x in num_cols if x in args.include_cols]

    md = model_dict[args.task_name][args.model_name]

    if args.model_name in ["rf", "elastic", "extra_trees", "xgb", "lgb"]:
        md["params"]["random_state"] = args.random_seed
    if args.model_name == "catboost":
        md["params"]["random_seed"] = args.random_seed
    for k in params_override:
        md["params"][k] = params_override[k]

    if args.task_name == "classification":
        cv = StratifiedKFold(5, shuffle=True, random_state=args.random_seed)
        scoring = "f1_score"
        class_conversion = {k: i for i, k in enumerate(np.sort(np.unique(y)))}
        class_conversion_rev = {
            class_conversion[k]: k for k in class_conversion
        }
    elif args.task_name == "regression":
        cv = KFold(5, shuffle=True, random_state=args.random_seed)
        scoring = "neg_root_mean_squared_error"
        class_conversion = None
        class_conversion_rev = None

    for train_idxs, val_idxs in kf.split(unique_study_uids, groups=groups):
        if groups is not None:
            f_curr = groups_dict[unique_study_uids[val_idxs[0]]]
        else:
            f_curr = f + 1
        print("Fold {}".format(f_curr))
        # data splitting
        train_uids = [unique_study_uids[i] for i in train_idxs]
        val_uids = [unique_study_uids[i] for i in val_idxs]
        train_idxs_long = np.where(study_uids.is_in(train_uids))[0]
        val_idxs_long = np.where(study_uids.is_in(val_uids))[0]
        training_X = X[train_idxs_long]
        training_y = y[train_idxs_long]
        val_X = X[val_idxs_long]
        val_y = y[val_idxs_long]

        print("\tTransforming data")
        if args.model_name != "catboost":
            count_vec = TextColsToCounts(
                text_cols={
                    i: x for i, x in enumerate(X.columns) if x in text_sep_cols
                },
                text_num_cols={
                    i: x for i, x in enumerate(X.columns) if x in num_sep_cols
                },
                num_cols={
                    i: x for i, x in enumerate(X.columns) if x in num_cols
                },
            )
            count_vec.fit(training_X)
            out = count_vec.transform(training_X)
            out_val = count_vec.transform(val_X)
            if model_name not in ["xgb", "lgb"]:
                out, training_y = nan_remover.transform(out, training_y)
                out_val, val_y = nan_remover.transform(out_val, val_y)
            if args.task_name == "classification":
                training_y = [class_conversion[x] for x in training_y]
                val_y = [class_conversion[x] for x in val_y]

            print(f"\tTraining model {model_name}. Data with shape={out.shape}")
            print(f"\t\tHyperparameters: {md['params']}")
            p = Pipeline(
                [
                    ("rzv", VarianceThreshold()),
                    ("model", md["model"](**md["params"])),
                ]
            )

            if len(md["cv_params"]) > 0:
                model = RandomizedSearchCV(
                    p,
                    {
                        "model__" + k: md["cv_params"][k]
                        for k in md["cv_params"]
                    },
                    verbose=3,
                    n_jobs=args.n_jobs,
                    cv=cv,
                    scoring=scoring,
                    n_iter=N_ITER,
                )
                model.fit(out, training_y)
                model = model.best_estimator_
            else:
                model = p
                model.fit(out, training_y)

            y_pred = model.predict(out_val)
            if args.task_name == "classification":
                val_y = [class_conversion_rev[x] for x in val_y]
                y_pred = [class_conversion_rev[x] for x in y_pred]
            cols_to_save = count_vec.new_col_names_

        else:
            cat_feature_cols = text_sep_cols + num_sep_cols
            fc = [x for x in training_X.columns if x in cat_feature_cols]
            training_data = Pool(
                data=FeaturesData(
                    cat_feature_data=np.array(training_X[fc]),
                    num_feature_data=np.array(
                        training_X[num_cols], dtype=np.float32
                    ),
                    num_feature_names=num_cols,
                    cat_feature_names=fc,
                ),
                label=np.array(training_y),
            )
            out_val = Pool(
                data=FeaturesData(
                    cat_feature_data=np.array(val_X[fc]),
                    num_feature_data=np.array(
                        val_X[num_cols], dtype=np.float32
                    ),
                    num_feature_names=num_cols,
                    cat_feature_names=fc,
                ),
                label=val_y,
            )
            val_y = out_val.get_label()

            print(
                f"\tTraining model {model_name} with {len(training_X)} samples"
            )
            model = md["model"](**md["params"])
            if len(md["cv_params"]) == 0:
                model.fit(training_data)
            else:
                print(f"\t\tHyperparameters: {md['params']}")
                model.randomized_search(
                    md["cv_params"],
                    X=training_data,
                    verbose=1,
                    cv=cv,
                    n_iter=N_ITER,
                )
            count_vec = None
            y_pred = model.predict(out_val)
            cols_to_save = num_cols + fc

        metric_dict = {
            "count_vec": count_vec,
            "model": model,
            "y_true": val_y,
            "y_pred": y_pred,
            "feature_names": cols_to_save,
            "fold": f_curr,
        }
        if args.task_name == "classification":
            metric_dict["auc"] = roc_auc_score(
                val_y,
                model.predict_proba(out_val),
                multi_class="ovo",
                labels=np.sort(np.unique(y)),
            )
        elif args.task_name == "regression":
            metric_dict["r2"] = r2_score(val_y, y_pred)
            print(f"\t\t{metric_dict['r2']}")
        train_results["cv"].append(metric_dict)

        f += 1

    print("Evaluating models (test dataset)")
    if args.test_set_path is not None:
        X_ho, y_ho, _, _ = data_loading_wraper(
            args.test_set_path,
            target_column=args.target_column,
            task=args.task_name,
        )
        train_results["test"] = []
        if model_name != "catboost":
            for i, f in enumerate(train_results["cv"]):
                count_vec = f["count_vec"]
                model = f["model"]
                test_data_pred = count_vec.transform(X_ho)
                if model_name not in ["xgb", "lgb"]:
                    test_data_pred, y_test = nan_remover.transform(
                        test_data_pred, y_ho
                    )
                else:
                    y_test = y_ho
                # calculate shape values
                explainer = shap.TreeExplainer(
                    model["model"], n_jobs=args.n_jobs
                )
                if args.skip_feature_importance is True:
                    feature_importance = None
                else:
                    feature_importance = np.stack(
                        explainer(
                            model["rzv"].transform(test_data_pred),
                            check_additivity=False,
                        ).values,
                        axis=1,
                    )

                # predict
                y_pred = model.predict(test_data_pred)
                if args.task_name == "classification":
                    y_pred = [class_conversion_rev[j] for j in y_pred]
                metric_dict = {
                    "y_true": y_test,
                    "y_pred": y_pred,
                    "feature_importance": feature_importance,
                    "fold": f["fold"],
                }
                if args.task_name == "classification":
                    metric_dict["auc"] = roc_auc_score(
                        y_test,
                        model.predict_proba(test_data_pred),
                        multi_class="ovr",
                    )
                elif args.task_name == "regression":
                    metric_dict["r2"] = r2_score(y_test, y_pred)
                train_results["test"].append(metric_dict)
        else:
            test_data_pred = Pool(
                data=FeaturesData(
                    cat_feature_data=np.array(X_ho[fc]),
                    num_feature_data=np.array(X_ho[num_cols], dtype=np.float32),
                    num_feature_names=num_cols,
                    cat_feature_names=fc,
                ),
                label=y_ho,
            )

            for i, f in enumerate(train_results["cv"]):
                model = f["model"]
                # calculate shap values
                if args.skip_feature_importance is True:
                    feature_importance = None
                else:
                    feature_importance = model.get_feature_importance(
                        data=test_data_pred, type=EFstrType.ShapValues
                    )

                y_pred = model.predict(test_data_pred)
                y_test = test_data_pred.get_label()
                metric_dict = {
                    "y_true": y_test,
                    "y_pred": y_pred,
                    "feature_importance": feature_importance,
                    "fold": f["fold"],
                }
                if args.task_name == "classification":
                    metric_dict["auc"] = roc_auc_score(
                        y_test,
                        model.predict_proba(test_data_pred),
                        multi_class="ovr",
                    )
                elif args.task_name == "regression":
                    metric_dict["r2"] = r2_score(y_test, y_pred)
                train_results["test"].append(metric_dict)

    # models can get quite heavy, so if you are just playing around
    # it might be best to skip saving the model
    if args.save_model != True:
        for i in range(len(train_results["cv"])):
            del train_results["cv"][i]["model"]

    # save everything
    Path(args.output_path).parent.mkdir(parents=True, exist_ok=True)

    with open(args.output_path, "wb") as f:
        dill.dump(train_results, f)
