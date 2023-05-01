import numpy as np
from sklearn.utils import check_random_state

class RandomForest:
    def __init__(self, n_estimators=10, max_depth=None, min_samples_split=2, base_estimator=DecisionTreeClassifier, random_state=None):
        # Initialize the class with the number of trees,
        # optional parameters for the base estimator, and a random seed
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.base_estimator = base_estimator
        self.random_state = random_state
        self.estimators = []
        
    def fit(self, X, y):
    # Build the decision trees using the training data (X and y)
    random_state = check_random_state(self.random_state)
    for i in range(self.n_estimators):
        # Sample a subset of the training data with replacement
        idx = random_state.randint(0, len(X), len(X))
        X_subset = X[idx]
        y_subset = y[idx]
        
        # Build a decision tree on the subset
        estimator = self.base_estimator(max_depth=self.max_depth, min_samples_split=self.min_samples_split)
        estimator.fit(X_subset, y_subset)
        self.estimators.append(estimator)
        
    def predict(self, X):
        # Use the built trees to make predictions on new data (X)
        predictions = []
        for estimator in self.estimators:
            predictions.append(estimator.predict(X))
            predictions = np.stack(predictions, axis=1)
            if self.base_estimator == DecisionTreeClassifier:
                # Return the majority class
                return np.apply_along_axis(lambda x: np.argmax(np.bincount(x)), axis=1, arr=predictions)
            else:
                # Return the average value
                return np.mean(predictions, axis=1