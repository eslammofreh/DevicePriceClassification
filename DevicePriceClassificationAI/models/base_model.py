from abc import ABC, abstractmethod
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from utils.visualization import Visualizer
import joblib
import os

class BaseModel(ABC):
    """
    Abstract base class for all models in the project.
    
    This class defines the common interface and shared methods for all models.
    """

    def __init__(self):
        """
        Initialize the BaseModel.
        """
        self.model = None
        self.visualizer = Visualizer()

    @abstractmethod
    def train(self, X, y):
        """
        Train the model.

        Args:
            X (array-like): The input samples.
            y (array-like): The target values.
        """
        pass

    @abstractmethod
    def predict(self, X):
        """
        Make predictions using the trained model.

        Args:
            X (array-like): The input samples.

        Returns:
            array-like: The predicted values.
        """
        pass

    def evaluate(self, X, y):
        y_pred = self.predict(X)
        
        accuracy = accuracy_score(y, y_pred)
        precision = precision_score(y, y_pred, average='weighted')
        recall = recall_score(y, y_pred, average='weighted')
        f1 = f1_score(y, y_pred, average='weighted')
        cm = confusion_matrix(y, y_pred)
        report = classification_report(y, y_pred)
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'confusion_matrix': cm,
            'classification_report': report
        }

    def plot_confusion_matrix(self, y_true, y_pred, class_names=None):
        """
        Plot the confusion matrix using the Visualizer.

        Args:
            y_true (array-like): True labels.
            y_pred (array-like): Predicted labels.
            class_names (list): List of class names (optional).
        """
        self.visualizer.plot_confusion_matrix(y_true, y_pred, class_names)

    def save_model(self, filename):
        """
        Save the trained model to a file.

        Args:
            filename (str): Name of the file to save the model to.
        """
        joblib.dump(self.model, filename)
        print(f"Model saved to {filename}")

    @classmethod
    def load_model(cls, filename):
        """
        Load a trained model from a file.

        Args:
            filename (str): Name of the file to load the model from.

        Returns:
            BaseModel: An instance of the model with the loaded model.
        """
        instance = cls()
        if os.path.exists(filename):
            instance.model = joblib.load(filename)
            print(f"Model loaded from {filename}")
        return instance