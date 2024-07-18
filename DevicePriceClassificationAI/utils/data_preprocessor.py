import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import pickle
import os 
from config import Config

class DataPreprocessor:
    def __init__(self):
        """
        Initialize the DataPreprocessor class.
        """
        self.preprocessor = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    def load_data(self, file_path):
        """
        Load the dataset from a CSV file.

        Args:
            file_path (str): Path to the CSV file.

        Returns:
            pd.DataFrame: Loaded dataset.
        """
        # Read the CSV file
        data = pd.read_csv(file_path)
        
        # Display basic information about the dataset
        print(Config.DATASET_INFO_MSG)
        print(data.info())
        
        # Display summary statistics
        print(Config.SUMMARY_STAT_MSG)
        print(data.describe())
        
        return data

    def split_data(self, data, test_size=0.2, random_state=42):
        """
        Split the data into training and testing sets.

        Args:
            data (pd.DataFrame): Input dataset.
            test_size (float): Proportion of the dataset to include in the test split.
            random_state (int): Random state for reproducibility.

        Returns:
            tuple: X_train, X_test, y_train, y_test
        """
        # Separate features and target
        X = data.drop('price_range', axis=1)
        y = data['price_range']
        
        # Split the data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        print(f"Training set shape: {self.X_train.shape}")
        print(f"Testing set shape: {self.X_test.shape}")
        
        return self.X_train, self.X_test, self.y_train, self.y_test

    def handle_missing_values(self, data):
        """
        Handle missing values in the dataset.

        Args:
            data (pd.DataFrame): Input dataset.

        Returns:
            pd.DataFrame: Dataset with missing values handled.
        """
        # Check for missing values
        missing_values = data.isnull().sum()
        print(Config.MISS_VALUES_MSG)
        print(missing_values[missing_values > 0])
        
        # If there are missing values, we'll impute them
        if missing_values.sum() > 0:
            numeric_features = data.select_dtypes(include=['int64', 'float64']).columns
            categorical_features = data.select_dtypes(include=(['object'])).columns
            
            # Impute numeric features with median
            if len(numeric_features) > 0:
                data[numeric_features] = data[numeric_features].fillna(data[numeric_features].median())
            
            # Impute categorical features with mode
            if len(categorical_features) > 0:
                data[categorical_features] = data[categorical_features].fillna(data[categorical_features].mode().iloc[0])
            
        return data

    def handle_outliers(self, data, columns, method='IQR'):
        """
        Handle outliers in specified columns.

        Args:
            data (pd.DataFrame): Input dataset.
            columns (list): List of columns to check for outliers.
            method (str): Method to use for outlier detection ('IQR' or 'zscore').

        Returns:
            pd.DataFrame: Dataset with outliers handled.
        """
        for column in columns:
            if method == Config.OUTLIER_TECHS['IQR']:
                Q1 = data[column].quantile(0.25)
                Q3 = data[column].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                data[column] = np.clip(data[column], lower_bound, upper_bound)
            elif method == Config.OUTLIER_TECHS['ZSCORE']:
                z_scores = np.abs((data[column] - data[column].mean()) / data[column].std())
                data[column] = data[column].mask(z_scores > 3, data[column].mean())
        
        return data

    def create_preprocessor(self):
        """
        Create a preprocessing pipeline.

        Returns:
            ColumnTransformer: Preprocessing pipeline.
        """

        # Create preprocessing steps for numeric features
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])

        # Create preprocessing steps for binary features
        binary_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent'))
        ])

        # Combine preprocessing steps
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, Config.NUMERIC_FEATURES),
                ('bin', binary_transformer, Config.BINARY_FEATURES)
            ])
        
        self.preprocessor = preprocessor
        
        return self.preprocessor

    @staticmethod
    def add_engineered_features(X):
        """
        Add engineered features to the dataset.

        Args:
            X (pd.DataFrame): Input features.

        Returns:
            pd.DataFrame: Features with added engineered features.
        """
        # Calculate screen area
        X['screen_area'] = X['sc_h'] * X['sc_w']
        
        # Calculate pixel density
        X['pixel_density'] = (X['px_height'] * X['px_width']) / X['screen_area']
        
        # Calculate battery efficiency
        X['battery_efficiency'] = X['battery_power'] / X['mobile_wt']
        
        # Calculate processor speed per core
        X['speed_per_core'] = X['clock_speed'] / X['n_cores']
        
        # Calculate memory per core
        X['memory_per_core'] = X['ram'] / X['n_cores']
        
        # Create a feature for total camera megapixels
        X['total_camera_mp'] = X['pc'] + X['fc']
        
        # Create interaction features
        X['ram_battery_interaction'] = X['ram'] * X['battery_power']
        
        print(Config.ENG_FEATURES_ADDED_MSG)
        print(X.columns[-9:].tolist())
        
        return X
    
    def get_numeric_columns(self, data):
        """
        Get the numeric columns from the dataset.

        Args:
            data (pd.DataFrame): Input dataset.

        Returns:
            list: List of numeric column names.
        """
        return data.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    def get_categorical_columns(self, data):
        
        """
        Get the categorical columns from the dataset.

        Args:
            data (pd.DataFrame): Input dataset.

        Returns:
            list: List of categorical column names.
        """
        return data.select_dtypes(include=['object']).columns.tolist()
    
    def load_preprocessor(filename):
        """Loads the preprocessor from a pickle file.

        Args:
            filename: The name of the file to load the preprocessor from.

        Returns:
            The loaded preprocessor object.
        """
        preprocessor = None
        if os.path.exists(filename):
            with open(filename, 'rb') as f:
                preprocessor = pickle.load(f)
        return preprocessor

    def save_preprocessor(self, preprocessor, filename):
        """Saves the preprocessor to a pickle file.

        Args:
            preprocessor: The preprocessor object to save.
            filename: The name of the file to save the preprocessor to.
        """

        os.makedirs(os.path.dirname(filename), exist_ok=True)
        pickle.dump(preprocessor, open(filename, "wb"))

    @staticmethod
    def handle_infinite_values(data):
        """
        Replace infinite values with NaN and then with the column median.
        """
        # Replace inf with NaN
        data = data.replace([np.inf, -np.inf], np.nan)
        
        # Replace NaN with column median
        for column in data.columns:
            data[column] = data[column].fillna(data[column].median())
        
        return data