from .base_model import BaseModel
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import StackingClassifier
from sklearn.model_selection import GridSearchCV
from config import Config

class StackingModel(BaseModel):
    """
    Stacking model for classification tasks.
    
    This class implements a stacking ensemble method using RandomForest and
    GradientBoosting as base models, and LogisticRegression as the final estimator.
    """

    def __init__(self, random_state=42):
        """
        Initialize the StackingModel.

        Args:
            random_state (int): Random state for reproducibility.
        """
        super().__init__()
        
        base_models = [
            ('rf', RandomForestClassifier(random_state=random_state)),
            ('gb', GradientBoostingClassifier(random_state=random_state))
        ]
        
        final_estimator = LogisticRegression(random_state=random_state)
        
        self.model = StackingClassifier(
            estimators=base_models,
            final_estimator=final_estimator,
            cv=Config.CROSS_VALIDATION
        )

    def train(self, X, y):
        """
        Train the Stacking model with hyperparameter tuning.

        Args:
            X (array-like): The input samples.
            y (array-like): The target values.
        """
        param_grid = {
            'rf__n_estimators': [100, 200],
            'rf__max_depth': [None, 10, 20],
            'gb__n_estimators': [100, 200],
            'gb__max_depth': [3, 5],
            'final_estimator__C': [0.1, 1, 10]
        }
        
        grid_search = GridSearchCV(self.model, param_grid, cv=5, n_jobs=-1, verbose=2)
        grid_search.fit(X, y)
        
        self.model = grid_search.best_estimator_
        print(Config.BEST_PARAMS_MSG, grid_search.best_params_)

    def predict(self, X):
        """
        Make predictions using the trained Stacking model.

        Args:
            X (array-like): The input samples.

        Returns:
            array-like: The predicted values.
        """
        return self.model.predict(X)

    def feature_importance(self, feature_names):
        """
        Get feature importance from the Stacking model.
        
        Note: This method returns feature importance only for the RandomForest
        base model in the stacking ensemble.

        Args:
            feature_names (list): List of feature names.

        Returns:
            dict: A dictionary mapping feature names to their importance scores.
        """
        rf_model = self.model.named_estimators_['rf']
        importances = rf_model.feature_importances_
        return dict(zip(feature_names, importances))

    def plot_feature_importance(self, feature_names):
        """
        Plot feature importance for the Stacking model using the Visualizer.
        
        Note: This method plots feature importance only for the RandomForest
        base model in the stacking ensemble.

        Args:
            feature_names (list): List of feature names.
        """
        rf_model = self.model.named_estimators_['rf']
        importances = rf_model.feature_importances_
        self.visualizer.plot_feature_importance(importances, feature_names)