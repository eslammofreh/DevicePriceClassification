from config import Config
from models.model_evaluation import ModelEvaluator
import joblib
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
import numpy as np

class ModelUtils:

    @staticmethod
    def evaluate_model(model, X_test_processed, y_test, model_name):
        """Evaluates a model and prints results."""
        evaluation = model.evaluate(X_test_processed, y_test)

        print(f"\n{model_name}" + Config.EVALUATION_MSG)
        for metric, value in evaluation.items():
            if type(value) == np.float64: 
                print(f"{metric.capitalize()}: {value:.4f}")
            else:
                print(f"{metric.capitalize()}: {value}")

        print(Config.CLASS_REPORT_MSG)
        print(evaluation[Config.EVAL_TECHS['CLASSIFICATION_REPORT']])

        return evaluation
    
    @staticmethod
    def visualize_model(model, X_test_processed, y_test, columns, visualizer):
        """Visualizes a model's performance."""
        visualizer.plot_confusion_matrix(y_test, model.predict(X_test_processed), class_names=Config.CLASSES.values())
        model.plot_feature_importance(columns)

    @staticmethod
    def interpret_model(model, X_test_processed):
        """Interprets a model using SHAP."""
        ModelEvaluator.interpret_with_shap(model.model, X_test_processed)

    @staticmethod
    def compare_models(model1_eval, model2_eval, primary_metric):
        """Compares two models based on given metric.

        Args:
            model1_eval: Evaluation results of the first model.
            model2_eval: Evaluation results of the second model.
            primary_metric: The primary metric for comparison.

        Returns:
            The better model based on the primary metric.
        """

        model1_score = model1_eval[primary_metric]
        model2_score = model2_eval[primary_metric]

        if model1_score > model2_score:
            return Config.MODELS['RANDOM_FOREST']
        elif model1_score < model2_score:
            return Config.MODELS['STACKING']
        
    def plot_tree(self, tree_index=0, feature_names=None, class_names=None):
        plt.figure(figsize=(20,10))
        plot_tree(self.model.estimators_[tree_index], 
                  feature_names=feature_names, 
                  class_names=class_names, 
                  filled=True, 
                  rounded=True)
        plt.title(f"Decision Tree {tree_index} from Random Forest")
        plt.show()