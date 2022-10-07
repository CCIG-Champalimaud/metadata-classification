from sklearn.ensemble import RandomForestClassifier,ExtraTreesClassifier
from sklearn.linear_model import SGDClassifier
from skopt.space import Real, Integer
from xgboost import XGBClassifier
from catboost import CatBoostClassifier,FeaturesData,Pool

model_dict = {
    "rf":{
        "model":RandomForestClassifier,
        "params":{"max_features":'sqrt'},
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
        "params":{"objective":"multi:softproba","n_jobs":2},
        "cv_params":{}
    },
    "catboost":{
        "model":CatBoostClassifier,
        "params":{"iterations":250,"verbose":False},
        "cv_params":{}
    }
}

model_dict_bayes = {
    "rf":{
        "model":RandomForestClassifier,
        "params":{},
        "cv_params":{"max_depth":Integer(3,20),
                     "min_samples_split":Integer(2,10)}
    },
    "elastic":{
        "model":SGDClassifier,
        "params":{"penalty":"elasticnet"},
        "cv_params":{"l1_ratio":Real(0,1)}
    },
    "extra_trees":{
        "model":ExtraTreesClassifier,
        "params":{},
        "cv_params":{"max_depth":Integer(3,20),
                     "min_samples_split":Integer(2,10)}
    },
    "xgb":{
        "model":XGBClassifier,
        "params":{"objective":"multi:softproba","n_jobs":2},
        "cv_params":{}
    },
    "catboost":{
        "model":CatBoostClassifier,
        "params":{"verbose":False},
        "cv_params":{}
    }
}
