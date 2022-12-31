# File containing the definition of the classes ADecisionTree, DecisionTreeClassifier and DecisionTreeRegressor
from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from typing import Literal, Tuple, Union 
from pydantic import conint, constr

from utils.node import Node

def ft_label_encoder(arr: np.ndarray) -> np.ndarray:

    map_label_encode = {}
    arr_encode = arr.copy()

    idx = 0
    for label in np.unique(arr):
        arr_encode[arr_encode == label] = idx
        map_label_encode[label] = idx 
        idx += 1
    
    return arr_encode, map_label_encode

class ADecisionTree(ABC):
    @abstractmethod
    def fit(self, data: pd.DataFrame, target: constr(strict=True)):
        ...

    @abstractmethod
    def _fit(self, X: np.ndarray, y: np.ndarray):
        ...

    @abstractmethod
    def _build_tree(self, X: np.ndarray, y: np.ndarray, depth: conint(gt=0, lt=20)):
        ...

    @abstractmethod
    def _best_split(self, X: np.ndarray, y: np.ndarray):
        ...

    @abstractmethod
    def _split(self, X: np.ndarray, threshold: Union[float, int], feature_type: Literal["categorical", "numerical"]):
        ...
    
    @abstractmethod
    def _leaf_value(self, y: np.ndarray):
        ...

    @abstractmethod
    def _create_node(self, feature: constr(strict=True), threshold: Union[float, int], depth: conint(gt=0, lt=20), left: Node, right: Node):
        ...

    @abstractmethod
    def predict(self, X: np.ndarray):
        ...

    @abstractmethod
    def _predict(self, tree: Node, X: np.ndarray):
        ...


class BaseDecisionTree(ADecisionTree):    
    def __init__(self, max_depth: int, min_samples_split: int):
        # Initialize the class with optional parameters max_depth and min_samples_splitpass
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tree = None
        self.map_data_encode = {}
        self.data_dtypes = None


    def _encode_data(self, X, y):
        """_summary_

        Args:
            X (_type_): _description_
            y (_type_): _description_

        Returns:
            (np.ndarray, np.ndarray): Encoded data features and target as numpy arrays
                                      with dtype within float or int.
        """
        n_samples = X.shape[0]
        X_encode = y_encode = np.zeros((n_samples, 1), dtype=int)
        self.data_dtypes = {feat: data_type for feat, data_type in zip(X.columns, X.dtypes.tolist())}
        for idx, column in enumerate(X.columns):
            if X[column].dtype == 'O':
                # if dtype is of string/categorical
                arr_encode, map_encode = ft_label_encoder(X[column].values)
                self.map_data_encode[column] = map_encode
                X_encode = np.concatenate((X_encode, arr_encode.reshape(-1,1)), axis=1)
            else:
                self.map_data_encode[column] = None
                X_encode = np.concatenate((X_encode, X[column].to_numpy().reshape(-1,1)), axis=1)
        X_encode = X_encode[:,1:]

        y_encode, map_target_encode = ft_label_encoder(y.values)
        self.map_data_encode['target'] = map_target_encode
        y_encode = y_encode.astype(int)
        return X_encode, y_encode


    def fit(self, data: pd.DataFrame, target: constr(strict=True)) -> None:
        """_summary_

        Args:
            data (pd.DataFrame): dataframe representing the features and the target.
            target (str): name of the column being the target.
        """
        features = [c for c in data.columns if c != target]

        X = data[features]
        y = data[target]
        idx = 0
        y_encode = y.copy()
        X_encode, y_encode = self._encode_data(X, y)
        self.tree = self._fit(X_encode, y_encode)


    def _fit(self, X: np.ndarray, y: np.ndarray) -> Node:
        """ _description_ """
        # Build the decision tree using the training data (X and y)
        return self._build_tree(X, y, 0)
        

    def _build_tree(self, X: np.ndarray, y: np.ndarray, depth: int) -> Node:
        """_summary_

        Args:
            X (_type_): _description_
            y (_type_): _description_
            depth (_type_): _description_

        Returns:
            _type_: _description_
        """
        n_samples, n_features = X.shape
        n_classes = len(np.unique(y))
        
        # Check if we need to terminate the recursion
        if n_samples < self.min_samples_split or (self.max_depth is not None and depth >= self.max_depth):
            # Return the value for a leaf node (e.g. the majority class)
            return self._leaf_value(y)
            
        # Check if all the samples belong to the same class
        if n_classes == 1:
            return self._create_node(None, None, 1.0, depth, True, None, None)
            
        # Select the best split feature and split value using some heuristic (e.g. information gain or gini impurity)
        best_feature, best_threshold, best_gain = self._best_split(X, y)
        
        # Split the data into two subsets
        best_feature_dtype = self.data_dtypes[list(self.data_dtypes.keys())[best_feature]]
        if best_feature_dtype.kind in ['f', 'i']:
            best_feature_dtype = "numerical"
        else:
            best_feature_dtype = "categorical"
        left_idx, right_idx = self._split(X[:, best_feature], best_threshold, feature_type=best_feature_dtype)
        X_left, X_right = X[left_idx], X[right_idx]
        y_left, y_right = y[left_idx], y[right_idx]
        
        # Recursively grow the left and right subtrees
        left_tree = self._build_tree(X_left, y_left, depth+1)
        right_tree = self._build_tree(X_right, y_right, depth+1)
        
        # Create and return a decision tree node
        return self._create_node(best_feature, best_threshold, best_gain, depth, False, left_tree, right_tree)
        

    def _best_split(self, X: np.ndarray, y: np.ndarray) -> Tuple[str, Union[float, int]]:
        """Select the best split feature and split value using some heuristic.
        
        Args:
            X (ndarray): The feature matrix.
            y (ndarray): The target vector.
        Returns:
            tuple: The best feature index and split value.
        """
        best_feature = None
        best_threshold = None
        best_gain = -np.inf
        
        # Loop over all the features
        key_data_dtypes = list(self.data_dtypes)
        for i in range(X.shape[1]):
            # Get the unique values for the feature
            unique_values = np.unique(X[:, i])
            
            # Loop over all the unique values (except the last one)
            if self.data_dtypes[key_data_dtypes[i]].kind in ['i', 'f']:
                for j in range(len(unique_values) - 1):
                    # Compute the split threshold
                    
                    threshold = (unique_values[j] + unique_values[j + 1]) / 2
                
                    # Split the data into two subsets
                    left_idx, right_idx = self._split(X[:, i], threshold, "numerical")
                    
                    X_left, X_right = X[left_idx], X[right_idx]
                    y_left, y_right = y[left_idx], y[right_idx]
                    # Compute the information gain
                    gain = self._information_gain(y, y_left, y_right)

                    # Update the best split if necessary
                    if gain > best_gain:
                        best_feature = i
                        best_threshold = threshold
                        best_gain = gain
            else:
                for threshold in unique_values:
                    # Split the data into two subsets
                    left_idx, right_idx = self._split(X[:, i], threshold, "categorical")
                    X_left, X_right = X[left_idx], X[right_idx]
                    y_left, y_right = y[left_idx], y[right_idx]
                
                    # Compute the information gain
                    gain = self._information_gain(y, y_left, y_right)

                    # Update the best split if necessary
                    if gain > best_gain:
                        best_feature = i
                        best_threshold = threshold
                        best_gain = gain
        return best_feature, best_threshold, best_gain
        

    def _split(self, X: np.ndarray, threshold: int, feature_type: Literal["categorical", "numerical"]):
        """Return the indices for the samples in the left and right subsets.
        
        Args:
            X (ndarray): The feature vector.
            threshold (float): The split threshold.
            threshold (str): Either "numerical" or "categorical". "numerical" if the datatype is
                        an integer or a float, "categorical" if the data type is a string or a
                        boolean.
        Returns:
            tuple: The indices for the samples in the left and right subsets.
        """
        if feature_type == "numerical":
            left_idx = np.where(X <= threshold)[0]
            right_idx = np.where(X > threshold)[0]
        elif feature_type == "categorical":
            left_idx = np.where(X == threshold)[0]
            right_idx = np.where(X != threshold)[0]
        return left_idx, right_idx


    def _create_node(self, feature: str, threshold: int, info_gain: float, depth: int, is_leaf: bool, left: Node, right: Node):
        """_summary_

        Args:
            feature (str): _description_
            threshold (_type_): _description_
            detph (int): _description_
            left (Node): _description_
            right (Node): _description_
        """
        # Create and return a decision tree node
        if is_leaf:
            return Node(feature, None, None, info_gain, depth+1, is_leaf=True)
        else:
            feat = list(self.map_data_encode.keys())[feature]
            if self.data_dtypes[feat].kind in ['i', 'f']:
                return Node(feat, "numerical", threshold, info_gain, depth+1, False, left=left, right=right)
            else:
                return Node(feat, "categorical", threshold, info_gain, depth+1, False, left=left, right=right)
    

    def predict(self, X: np.ndarray):
        # Use the built tree to make predictions on new data (X)
        return self._predict(self.tree, X)

    @abstractmethod
    def _predict(self, tree: Node, X: np.ndarray):
        """Use the decision tree to make predictions on new data (X).
        
        Args:
            tree (tuple): The decision tree.
            X (ndarray): The feature matrix.
            
        Returns:
            ndarray: The predicted target vector.
        """
        pass


class MyDecisionTreeRegressor(BaseDecisionTree):
    def __init__(self, max_depth=2, min_samples_split=2):
        super().__init__(max_depth, min_samples_split)


    def _leaf_value(self, y: np.ndarray):
        """Return the value for a leaf node (e.g. the majority class).
        
        Args:
            y (ndarray): The target vector.
        Returns:
            any: The value for a leaf node.
        """
        return np.mean(y)


    def _predict(self, tree: Node, X: np.ndarray):
        """Use the decision tree to make predictions on new data (X).
        
        Args:
            tree (tuple): The decision tree.
            X (ndarray): The feature matrix.
            
        Returns:
            ndarray: The predicted target vector.
        """
        if tree is None:
            return None
            
        if tree.feature is None:
            return tree.right # plutot utiliser la maniere avec la classe Node
            
        feature = tree.feature
        threshold = tree.threshold
        left = tree.left
        right = tree.right
        
        if self.data_dtypes[feature] == "categorical":
            if X[feature] == threshold:
                return self._predict(left, X)
            else:
                return self._predict(right, X)
        elif self.data_dtypes[feature] == "numerical":
            if X[feature] <= threshold:
                return self._predict(left, X)
            else:
                return self._predict(right, X)
