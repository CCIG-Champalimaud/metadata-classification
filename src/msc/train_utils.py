import importlib
from dataclasses import dataclass
from scipy.stats import loguniform, randint


@dataclass
class ModelConstructor:
    task_type: str
    """
    Class to construct models.
        
    Args:
        task_type (str): Type of task (classification or regression).
    """

    def import_models(self, lib: str, model_names: list[str]):
        """
        Convenience function to import models using exec.
        """
        try:
            module = importlib.import_module(lib)
            models = []
            for model_name in model_names:
                models.append(getattr(module, model_name))
            return models
        except ModuleNotFoundError:
            lib = lib.split(".")[0]
            raise ImportError(f"Could not import {lib} - please install it.")

    @property
    def model_dict(self):
        """
        Model import dictionary.
        """
        return {
            "rf": {
                "lib": "sklearn.ensemble",
                "models": ["RandomForestClassifier", "RandomForestRegressor"],
            },
            "extra_trees": {
                "lib": "sklearn.ensemble",
                "models": ["ExtraTreesClassifier", "ExtraTreesRegressor"],
            },
            "elastic": {
                "lib": "sklearn.linear_model",
                "models": ["SGDClassifier", "SGDRegressor"],
            },
            "xgb": {
                "lib": "xgboost",
                "models": ["XGBClassifier", "XGBRegressor"],
            },
            "lgbm": {
                "lib": "lightgbm",
                "models": ["LGBMClassifier", "LGBMRegressor"],
            },
            "catboost": {
                "lib": "catboost",
                "models": ["CatBoostClassifier", "CatBoostRegressor"],
            },
        }

    def __call__(self, model_name: str):
        """
        Returns the constructor for a given model name and task type.

        Args:
            model_name (str): Name of the model.

        Returns:
            Callable: Constructor for the model.
        """

        if model_name in self.model_dict:
            lib, models = self.model_dict[model_name].values()
            models = self.import_models(lib, models)
            if self.task_type == "classification":
                return models[0]
            elif self.task_type == "regression":
                return models[1]
        else:
            raise ValueError(f"Unknown model name: {model_name}")


model_dict = {
    "classification": {
        "rf": {
            "params": {"max_features": "sqrt", "n_estimators": 50},
            "cv_params": {
                "max_depth": [None, 10, 20],
                "min_samples_split": randint(2, 8),
            },
        },
        "elastic": {
            "params": {"penalty": "elasticnet", "loss": "modified_huber"},
            "cv_params": {"l1_ratio": loguniform(0.1, 0.99)},
        },
        "extra_trees": {
            "params": {"max_features": "sqrt"},
            "cv_params": {
                "max_depth": [None, 10, 20],
                "min_samples_split": randint(2, 8),
            },
        },
        "xgb": {
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
            "params": {"max_features": "sqrt", "n_estimators": 50},
            "cv_params": {
                "max_depth": [None, 10, 20],
                "min_samples_split": randint(2, 8),
            },
        },
        "elastic": {
            "params": {"penalty": "elasticnet", "loss": "modified_huber"},
            "cv_params": {"l1_ratio": loguniform(0.1, 0.99)},
        },
        "extra_trees": {
            "params": {"max_features": "sqrt"},
            "cv_params": {
                "max_depth": [None, 10, 20],
                "min_samples_split": randint(2, 8),
            },
        },
        "xgb": {
            "params": {
                "objective": "reg:squarederror",
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
