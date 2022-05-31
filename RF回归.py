# -*- coding: utf-8 -*-
"""
Created on Tue May 31 16:28:57 2022

@author: HZAUerhanshen
"""

from sklearn.datasets import load_iris,load_breast_cancer
import numpy as np
from sklearn.model_selection import train_test_split
from collections import Counter
import random
from sklearn.datasets import load_iris,load_breast_cancer,load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error,mean_squared_error
class Node:
    def __init__(self, mean,num_samples):
        self.num_samples = num_samples
        self.feature_index = 0
        self.threshold = 0
        self.left = None
        self.right = None
        self.mean=mean
class MetaLearner(object):
    def __init__(self,min_samples:int=4):
        self.max_depth =7
    def calEntropy(self,y):
        #print(y,len(y))
        var=np.var(y)
        #计算当前数据的经验熵
        return var
    def _best_split(self, X, y):
        m = y.size
        if m <= 10:
            return None, None
        dataset_Entropy =self.calEntropy(y)
        best_idx, best_thr = None, None
        best_gain_info=0
        for idx in range(self.n_features_):
            #产生候选的阈值
            thresholds=X[:,idx]
            #print(thresholds)
            for i in range(m):
                #根据阈值划分数据集，并计算子数据集的信息增益
                is_left_features=X[:,idx]>=thresholds[i]
                True_index=[i for i,x in enumerate(is_left_features) if x==True]
                suby=[]
                for j in True_index:
                    suby.append(y[j])
                var_left=self.calEntropy(suby)*float(len(suby))/m
                is_right_features=X[:,idx]<=thresholds[i]
                True_index=[i for i,x in enumerate(is_right_features) if x==True]
                suby=[]
                for j in True_index:
                    suby.append(y[j])
                var_right=self.calEntropy(suby)*float(len(suby))/m
                Gain_var=dataset_Entropy-(var_left+var_right)
                if thresholds[i] == thresholds[i - 1]:
                        continue
                if Gain_var>best_gain_info and Gain_var>=0.1:
                    #print(i,idx)
                    best_gain_info= Gain_var
                    best_thr = thresholds[i]
                    best_idx=idx
        return best_idx, best_thr
    def fit(self, X, y):
        self.n_features_ =X.shape[1]
        self.tree_ = self._grow_tree(X, y)

    def _grow_tree(self, X, y, depth=0):
        #print(y,len(y))
        node = Node(mean=np.mean(y),
            num_samples=y.size,)

        if self.max_depth is not None and depth < self.max_depth:
            idx, thr = self._best_split(X, y)
            if idx is not None:
                indices_left = X[:, idx] < thr
                X_left, y_left = X[indices_left], y[indices_left]
                X_right, y_right = X[~indices_left], y[~indices_left]
                node.feature_index = idx
                node.threshold = thr
                node.left = self._grow_tree(X_left, y_left, depth + 1)
                node.right = self._grow_tree(X_right, y_right, depth + 1)
        return node
    def predict(self, X):
        return np.array([self._predict(inputs) for inputs in X])

    def _predict(self, inputs):
        node = self.tree_
        while node.left:
            if inputs[node.feature_index] < node.threshold:
                node = node.left
            else:
                node = node.right
        return node.mean

def bootstrap_sample(X, y):
    n_samples = X.shape[0]
    idxs = np.random.choice(n_samples, n_samples, replace=True)
    #print(len(idxs))
    return X[idxs], y[idxs]
def most_common_label(y):
    counter = Counter(y)
    most_common = counter.most_common(1)[0][0]
    return most_common
class RandomForest:
    def __init__(self, n_trees=10,max_depth=7):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.trees = []

    def fit(self, X, y):
        self.trees = []
        for _ in range(self.n_trees):
            random.seed(_)
            tree = MetaLearner(10)
            X_samp, y_samp = bootstrap_sample(X, y)
            tree.fit(X_samp,y_samp)
            self.trees.append(tree)

    def predict(self, X):
        tree_preds = np.array([tree.predict(X) for tree in self.trees])
        tree_preds = np.swapaxes(tree_preds,0,1)
        #print(tree_preds)
        #y_pred = [most_common_label(tree_pred) for tree_pred in tree_preds]
        y_pred=[np.mean(i) for i in tree_preds]
        return np.array(y_pred)
iris=load_diabetes()
x_train, x_test, y_train, y_test = train_test_split(iris.data,iris.target,random_state=21)
RF=RandomForest(n_trees=5)
RF.fit(x_train,y_train)
y_pred=RF.predict(x_test)
print(mean_absolute_error(y_test,y_pred))
