import pandas as pd
from sklearn.model_selection import train_test_split

# Data Loading and Preprocessing
class DataLoader:
    """
    Class for loading and preprocessing the mobile device dataset.
    """
    def __init__(self, file_path):
        """
        Initialize the DataLoader.

        Args:
            file_path (str): Path to the CSV file containing the dataset.
        """
        self.file_path = file_path
        self.data = None

    def load_data(self):
        """
        Load the dataset from the CSV file.

        Returns:
            pd.DataFrame: Loaded dataset.
        """
        self.data = pd.read_csv(self.file_path)
        return self.data

    def split_data(self, test_size=0.2, random_state=42):
        """
        Split the dataset into training and testing sets.

        Args:
            test_size (float): Proportion of the dataset to include in the test split.
            random_state (int): Random state for reproducibility.

        Returns:
            tuple: X_train, X_test, y_train, y_test
        """
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        X = self.data.drop('price_range', axis=1)
        y = self.data['price_range']
        return train_test_split(X, y, test_size=test_size, random_state=random_state)
