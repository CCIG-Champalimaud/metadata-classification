import numpy as np
from copy import deepcopy
from sklearn.feature_extraction.text import CountVectorizer

class SpaceSepNumColsToMatrix:
    def __init__(self,standard=False):
        self.standard = standard

    def fit(self,X,y=None):
        mat = [x.strip().split(" ") for x in X]
        sizes,counts = np.unique([len(x) for x in mat],
                                 return_counts=True)
        if self.standard == False:
            self.transform_ = "sum_size"
            self.n_features_ = 5
            self.feature_names_ = ["length","sum","min","max","mean"]
        else:
            self.transform_ = "standard"
            self.n_features_ = sizes[0]
            self.feature_names_ = [i for i in range(self.n_features_)]
    
    def safe_feat(self,x):
        try:
            x = [i for i in x if i != ""]
            x = np.float32(x)
            return x.sum(),x.min(),x.max(),x.mean()
        except:
            return np.nan,np.nan,np.nan,np.nan

    def transform(self,X,y=None):
        mat = [x.strip().split(" ") for x in X]
        if self.transform_ == "standard":
            mat = np.array(mat).astype(np.float32)
            if mat.shape[1] != self.n_features_:
                raise Exception("different number of elements")
        elif self.transform_ == "sum_size":
            mat = np.array([[len(x),*self.safe_feat(x)]
                            for x in mat])
        return mat.astype(np.float32)
    
    def fit_transform(self,X,y=None):
        self.fit(X)
        return self.transform(X)

class TextColsToCounts:
    def __init__(self,
                 text_cols={},
                 text_num_cols={},
                 num_cols={}):
        self.text_cols = text_cols
        self.text_num_cols = text_num_cols
        self.num_cols = num_cols
    
    def fit(self,X,y=None):
        self.all_cols_ = sorted(
            [*self.num_cols,*self.text_cols,*self.text_num_cols])
        X = np.array(X)
        self.vectorizers_ = {}
        self.col_name_dict_ = {}
        for col in self.text_cols:
            self.vectorizers_[col] = CountVectorizer()
            d = X[:,col]
            self.vectorizers_[col].fit(d)
            self.col_name_dict_[col] = [
                "{}:{}".format(self.text_cols[col],x)
                for x in self.vectorizers_[col].vocabulary_]
        for col in self.text_num_cols:
            self.vectorizers_[col] = SpaceSepNumColsToMatrix()
            d = X[:,col]
            self.vectorizers_[col].fit(d)
            self.col_name_dict_[col] = [
                "{}:{}".format(self.text_num_cols[col],x)
                for x in self.vectorizers_[col].feature_names_]
        for col in self.num_cols:
            self.col_name_dict_[col] = [self.num_cols[col]]

        self.new_col_names_ = []
        for c in self.col_name_dict_:
            self.new_col_names_.extend(self.col_name_dict_[c])
    
    def transform(self,X,y=None):
        X = np.array(deepcopy(X))
        output = []
        for col in self.all_cols_:
            if col in self.text_cols:
                mat = self.vectorizers_[col].transform(X[:,col])
                mat = mat.todense()
            elif col in self.text_num_cols:
                mat = self.vectorizers_[col].transform(X[:,col])
            else:
                mat = np.array(X[:,col])[:,np.newaxis]
            output.append(np.array(mat))
        return np.concatenate(output,axis=1).astype(np.float32)

    def fit_transform(self,X,y=None):
        self.fit(X)
        return self.transform(X)

class RemoveNan:
    def __init__(self):
        pass

    def fit(self,X,y):
        return self

    def transform(self,X,y):
        X = deepcopy(X)
        idxs = np.isnan(X).sum(1) > 1
        X = X[~idxs]
        y = y[~idxs]
        return X,y
    
    def fit_transform(self,X,y):
        return self.transform(X,y)
