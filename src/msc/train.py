import argparse
import os
import numpy as np
import dill
import json
import fasttreeshap as shap
from sklearn.metrics import roc_auc_score
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cross validates models.")

    parser.add_argument(
        "--input_path", dest="input_path", type=str, required=True
    )
    parser.add_argument(
        "--test_set_path", dest="test_set_path", type=str, default=None
    )
    parser.add_argument(
        "--output_path", dest="output_path", type=str, required=True
    )
    parser.add_argument("--n_folds", dest="n_folds", type=int, default=10)
    parser.add_argument("--folds", dest="folds", type=str, default=None)
    parser.add_argument("--n_jobs", dest="n_jobs", type=int, default=0)
    parser.add_argument(
        "--model_name",
        dest="model_name",
        default="rf",
        choices=["rf", "elastic", "extra_trees", "xgb", "catboost"],
    )
    parser.add_argument("--exclude_cols", dest="exclude_cols", nargs="+")
    parser.add_argument("--save_model", dest="save_model", action="store_true")
    parser.add_argument(
        "--random_seed", dest="random_seed", default=42, type=int
    )
    parser.add_argument("--subset", dest="subset", default=None, type=float)
    parser.add_argument(
        "--skip_feature_importance",
        dest="skip_feature_importance",
        default=False,
        action="store_true",
    )
    args = parser.parse_args()

    X, y, study_uids, unique_study_uids = data_loading_wraper(args.input_path)
    if args.subset is not None:
        if args.subset > 1.0:
            subset = int(args.subset)
        else:
            subset = int(args.subset * X.shape[0])
        idxs = np.random.choice(X.shape[0], subset, replace=False)
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
        all_cols = text_sep_cols + num_sep_cols + num_cols

    class_conversion = {k: i for i, k in enumerate(np.sort(np.unique(y)))}
    class_conversion_rev = {class_conversion[k]: k for k in class_conversion}

    md = model_dict

    if args.model_name in ["rf", "elastic", "extra_trees", "xgb"]:
        md[args.model_name]["params"]["random_state"] = args.random_seed
    if args.model_name == "catboost":
        md[args.model_name]["params"]["random_seed"] = args.random_seed

    for train_idxs, val_idxs in kf.split(unique_study_uids, groups=groups):
        if groups is not None:
            f_curr = groups_dict[unique_study_uids[val_idxs[0]]]
        else:
            f_curr = f + 1
        print("Fold {}".format(f_curr))
        # data splitting
        train_uids = [unique_study_uids[i] for i in train_idxs]
        val_uids = [unique_study_uids[i] for i in val_idxs]
        train_idxs_long = [
            i for i, x in enumerate(study_uids) if x in train_uids
        ]
        val_idxs_long = [i for i, x in enumerate(study_uids) if x in val_uids]
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
            if model_name != "xgb":
                out, training_y = nan_remover.transform(out, training_y)
                out_val, val_y = nan_remover.transform(out_val, val_y)
            training_y = [class_conversion[x] for x in training_y]
            val_y = [class_conversion[x] for x in val_y]

            print(
                f"\tTraining model {model_name} with {len(training_X)} samples"
            )
            print(f"\t\tHyperparameters: {md[model_name]['params']}")
            p = Pipeline(
                [
                    ("rzv", VarianceThreshold()),
                    (
                        "model",
                        md[model_name]["model"](**md[model_name]["params"]),
                    ),
                ]
            )

            if len(md[model_name]["cv_params"]) > 0:
                cv = StratifiedKFold(
                    5, shuffle=True, random_state=args.random_seed
                )
                model = RandomizedSearchCV(
                    p,
                    {
                        "model__" + k: md[model_name]["cv_params"][k]
                        for k in md[model_name]["cv_params"]
                    },
                    verbose=1,
                    n_jobs=args.n_jobs,
                    cv=cv,
                    scoring="f1_macro",
                    n_iter=N_ITER,
                )
                model.fit(out, training_y)
                model = model.best_estimator_
            else:
                model = p
                model.fit(out, training_y)

            y_pred = model.predict(out_val)
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
            model = md[model_name]["model"](**model_dict[model_name]["params"])
            if len(model_dict[model_name]["cv_params"]) == 0:
                model.fit(training_data)
            else:
                print(
                    f"\t\tHyperparameters: {model_dict[model_name]['cv_params']}"
                )
                cv = StratifiedKFold(
                    5, shuffle=True, random_state=args.random_seed
                )
                model.randomized_search(
                    model_dict[model_name]["cv_params"],
                    X=training_data,
                    verbose=0,
                    cv=cv,
                    n_iter=N_ITER,
                )
            count_vec = None
            y_pred = model.predict(out_val)
            cols_to_save = num_cols + fc

        train_results["cv"].append(
            {
                "auc": roc_auc_score(
                    val_y,
                    model.predict_proba(out_val),
                    multi_class="ovo",
                    labels=np.sort(np.unique(y)),
                ),
                "count_vec": count_vec,
                "model": model,
                "y_true": val_y,
                "y_pred": y_pred,
                "feature_names": cols_to_save,
                "fold": f_curr,
            }
        )

        f += 1

    print("Evaluating models (test dataset)")
    if args.test_set_path is not None:
        X_ho, y_ho, _, _ = data_loading_wraper(args.test_set_path)
        train_results["test"] = []
        if args.model_name != "catboost":
            for i, f in enumerate(train_results["cv"]):
                count_vec = f["count_vec"]
                model = f["model"]
                test_data_pred = count_vec.transform(X_ho)
                if model_name != "xgb":
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

                train_results["test"].append(
                    {
                        "y_true": y_test,
                        "y_pred": [class_conversion_rev[j] for j in y_pred],
                        "feature_importance": feature_importance,
                        "auc": roc_auc_score(
                            y_test,
                            model.predict_proba(test_data_pred),
                            multi_class="ovr",
                        ),
                        "fold": f["fold"],
                    }
                )
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
                train_results["test"].append(
                    {
                        "y_true": y_test,
                        "y_pred": y_pred,
                        "feature_importance": feature_importance,
                        "auc": roc_auc_score(
                            y_test,
                            model.predict_proba(test_data_pred),
                            multi_class="ovr",
                        ),
                        "fold": f["fold"],
                    }
                )

    # models can get quite heavy, so if you are just playing around
    # it might be best to skip saving the model

    if args.save_model != True:
        for i in range(len(train_results["cv"])):
            del train_results["cv"][i]["model"]

    # save everything
    dir_name = os.path.dirname(args.output_path)
    os.makedirs(dir_name, exist_ok=True)

    with open(args.output_path, "wb") as f:
        dill.dump(train_results, f)
