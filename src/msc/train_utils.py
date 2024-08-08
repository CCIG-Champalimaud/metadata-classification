from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import SGDClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from scipy.stats import loguniform, randint

model_dict = {
    "rf": {
        "model": RandomForestClassifier,
        "params": {"max_features": "sqrt", "n_estimators": 50},
        "cv_params": {
            "max_depth": [None, 10, 20],
            "min_samples_split": [2, 4, 8],
        },
    },
    "elastic": {
        "model": SGDClassifier,
        "params": {"penalty": "elasticnet", "loss": "modified_huber"},
        "cv_params": {"l1_ratio": [0.1, 0.5, 0.9, 0.95, 0.99]},
    },
    "extra_trees": {
        "model": ExtraTreesClassifier,
        "params": {"max_features": "sqrt"},
        "cv_params": {
            "max_depth": [None, 10, 20],
            "min_samples_split": [2, 4, 8],
        },
    },
    "xgb": {
        "model": XGBClassifier,
        "params": {
            "objective": "multi:softproba",
            "n_jobs": 2,
            "tree_method": "gpu_hist",
            "predictor": "auto",
            "sampling_method": "gradient_based",
            "subsample": 0.5,
            "verbosity": 0,
        },
        "cv_params": {
            "learning_rate": loguniform(1e-4, 3e-1),
            "max_depth": randint(3, 10),
            "min_child_weight": randint(1, 5),
            "gamma": loguniform(1e-4, 1e-1),
        },
    },
    "catboost": {
        "model": CatBoostClassifier,
        "params": {
            "iterations": 1000,
            "verbose": False,
            "task_type": "GPU",
            "leaf_estimation_method": "Newton",
        },
        "cv_params": {
            "max_bin": randint(100, 255),
            "depth": randint(4, 9),
            "l2_leaf_reg": loguniform(1e-1, 1e1),
            "learning_rate": loguniform(1e-2, 3e-1),
        },
    },
}
