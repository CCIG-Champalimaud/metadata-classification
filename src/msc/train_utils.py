from sklearn.ensemble import RandomForestClassifier,ExtraTreesClassifier
from sklearn.linear_model import SGDClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier,FeaturesData,Pool

model_dict = {
    "rf":{
        "model":RandomForestClassifier,
        "params":{"max_features":'sqrt',"n_estimators":50},
        "cv_params":{"max_depth":[None,10,20],"min_samples_split":[2,4,8]}
    },
    "elastic":{
        "model":SGDClassifier,
        "params":{"penalty":"elasticnet","loss":"modified_huber"},
        "cv_params":{"l1_ratio":[0.1,0.5,0.9,0.95,0.99]}
    },
    "extra_trees":{
        "model":ExtraTreesClassifier,
        "params":{"max_features":'sqrt'},
        "cv_params":{"max_depth":[None,10,20],"min_samples_split":[2,4,8]}
    },
    "xgb":{
        "model":XGBClassifier,
        "params":{"objective":"multi:softproba",
                  "n_jobs":2,
                  "tree_method":"gpu_hist",
                  "predictor":"auto",
                  "sampling_method":"gradient_based",
                  "subsample":0.5,
                  "verbosity":0},
        "cv_params":{}
    },
    "catboost":{
        "model":CatBoostClassifier,
        "params":{"iterations":250,"verbose":False,
                  "task_type":"CPU","thread_count":4,
                  "leaf_estimation_method":"Gradient"},
        "cv_params":{}
    }
}
