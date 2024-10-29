from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    RandomForestRegressor,
    ExtraTreesRegressor,
)
from sklearn.linear_model import SGDClassifier, SGDRegressor
from xgboost import XGBClassifier, XGBRegressor
from catboost import CatBoostClassifier, CatBoostRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from scipy.stats import loguniform, randint

model_dict = {
    "classification": {
        "rf": {
            "model": RandomForestClassifier,
            "params": {"max_features": "sqrt", "n_estimators": 50},
            "cv_params": {
                "max_depth": [None, 10, 20],
                "min_samples_split": randint(2, 8),
            },
        },
        "elastic": {
            "model": SGDClassifier,
            "params": {"penalty": "elasticnet", "loss": "modified_huber"},
            "cv_params": {"l1_ratio": loguniform(0.1, 0.99)},
        },
        "extra_trees": {
            "model": ExtraTreesClassifier,
            "params": {"max_features": "sqrt"},
            "cv_params": {
                "max_depth": [None, 10, 20],
                "min_samples_split": randint(2, 8),
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
        "lgb": {
            "model": LGBMClassifier(),
            "params": {
                "objective": "multiclass",
                "n_jobs": 1,
                "boosting_type": "gbdt",
                "subsample": 0.5,
                "subsample_freq": 1,
                "verbose": 0,
                "max_depth": -1,
            },
            "cv_params": {
                "num_leaves": randint(31, 100),
                "learning_rate": loguniform(0.01, 0.1),
                "n_estimators": randint(10, 100),
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
    },
    "regression": {
        "rf": {
            "model": RandomForestRegressor,
            "params": {"max_features": "sqrt", "n_estimators": 50},
            "cv_params": {
                "max_depth": [None, 10, 20],
                "min_samples_split": randint(2, 8),
            },
        },
        "elastic": {
            "model": SGDRegressor,
            "params": {"penalty": "elasticnet", "loss": "modified_huber"},
            "cv_params": {"l1_ratio": loguniform(0.1, 0.99)},
        },
        "extra_trees": {
            "model": ExtraTreesRegressor,
            "params": {"max_features": "sqrt"},
            "cv_params": {
                "max_depth": [None, 10, 20],
                "min_samples_split": randint(2, 8),
            },
        },
        "xgb": {
            "model": XGBRegressor,
            "params": {
                "objective": "reg:tweedie",
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
        "lgb": {
            "model": LGBMRegressor,
            "params": {
                "objective": "regression",
                "n_jobs": 1,
                "boosting_type": "gbdt",
                "subsample": 0.5,
                "subsample_freq": 1,
                "verbose": 0,
                "max_depth": -1,
            },
            "cv_params": {
                "num_leaves": randint(31, 100),
                "learning_rate": loguniform(0.01, 0.1),
                "n_estimators": randint(10, 100),
            },
        },
        "catboost": {
            "model": CatBoostRegressor,
            "params": {
                "iterations": 1000,
                "verbose": False,
                "task_type": "GPU",
                "bootstrap_type": "Poisson",
                "leaf_estimation_method": "Newton",
            },
            "cv_params": {
                "max_bin": randint(100, 255),
                "depth": randint(4, 9),
                "l2_leaf_reg": loguniform(1e-1, 1e1),
                "learning_rate": loguniform(1e-2, 3e-1),
            },
        },
    },
}
