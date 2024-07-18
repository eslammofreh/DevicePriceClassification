from .base_model import BaseModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from config import Config
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

class RandomForestModel(BaseModel):
    """
    Random Forest model for classification tasks.
    
    This class implements the Random Forest algorithm and includes
    hyperparameter tuning using GridSearchCV.
    """

    def __init__(self, **kwargs):
        """
        Initialize the RandomForestModel.

        Args:
            n_estimators (int): The number of trees in the forest.
            max_depth (int): The maximum depth of the tree.
            min_samples_split (int): The minimum number of samples required to split an internal node.
            min_samples_leaf (int): The minimum number of samples required to be at a leaf node.
            max_features (str): The number of features to consider when looking for the best split.
            random_state (int): Random state for reproducibility.
        """
        super().__init__()
        self.model = RandomForestClassifier(**kwargs)

    def train(self, X, y):
        """
        Train the Random Forest model with hyperparameter tuning.

        Args:
            X (array-like): The input samples.
            y (array-like): The target values.
        """
        grid_search = GridSearchCV(self.model, Config.RANDOM_FOREST_PARAMS, cv=5, n_jobs=-1, verbose=2)
        grid_search.fit(X, y)
        self.model = grid_search.best_estimator_
        print(Config.BEST_PARAMS_MSG, grid_search.best_params_)

    def predict(self, X):
        """
        Make predictions using the trained Random Forest model.

        Args:
            X (array-like): The input samples.

        Returns:
            array-like: The predicted values.
        """
        return self.model.predict(X)

    def feature_importance(self, feature_names):
        """
        Get feature importance from the Random Forest model.

        Args:
            feature_names (list): List of feature names.

        Returns:
            dict: A dictionary mapping feature names to their importance scores.
        """
        importances = self.model.feature_importances_
        if len(importances) != len(feature_names):
            raise ValueError(Config.MISSMATCH_FEATURE_IMPORT_NAMES_MSG)
        return dict(zip(feature_names, importances))

    def plot_feature_importance(self, feature_names):
        """
        Plot feature importance for the Random Forest model using the Visualizer.

        Args:
            feature_names (list): List of feature names.
        """
        importances = self.model.feature_importances_
        self.visualizer.plot_feature_importance(importances, feature_names)

    def plot_tree(self, tree_index=0, feature_names=None, class_names=None):
        """
        Plot a single tree from the Random Forest.

        Args:
            tree_index (int): Index of the tree to plot.
            feature_names (list): List of feature names.
            class_names (list): List of class names.
        """    
        plt.figure(figsize=(20,10))
        plot_tree(self.model.estimators_[tree_index], 
                  feature_names=feature_names, 
                  class_names=class_names, 
                  filled=True, 
                  rounded=True)
        plt.title(f"Decision Tree {tree_index} from Random Forest")
        plt.show()