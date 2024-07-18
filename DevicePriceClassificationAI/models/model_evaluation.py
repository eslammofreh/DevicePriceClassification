import numpy as np
from sklearn.metrics import confusion_matrix
import shap
  
class ModelEvaluator:
    """Class for model evaluation and interpretation."""
    @staticmethod
    def interpret_with_shap(model, X):
        """
        Interpret the model using SHAP values.

        Args:
            model: Trained model.
            X (np.array): Input features.
        """
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)
        shap.summary_plot(shap_values, X, plot_type="bar")