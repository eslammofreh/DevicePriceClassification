import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os
sys.path.append('../') 
from config import Config
from visualization import Visualizer, DataExplorer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

class EDA:
    def __init__(self, data_path):
        """
        Initialize the EDA class.

        Args:
            data_path (str): Path to the dataset.
        """
        self.data = pd.read_csv(data_path)
        self.visualizer = Visualizer(output_dir="../results/data_exploration_results")
        self.data_explorer = DataExplorer(self.data, visualizer="../results/data_exploration_results")

    def preprocess_data(self):
        """
        Preprocess the data by handling missing values and encoding categorical variables.
        """
        # Handle missing values
        self.data = self.data.fillna(self.data.mean())

        # Encode binary features
        for feature in Config.BINARY_FEATURES:
            self.data[feature] = self.data[feature].astype(int)


    def perform_eda(self):
        """
        Perform Exploratory Data Analysis.
        """
        print(Config.DATASET_INFO_MSG)
        print(self.data.info())

        print(Config.SUMMARY_STAT_MSG)
        print(self.data.describe())

        print(Config.MISS_VALUES_MSG)
        print(self.data.isnull().sum())

        # Visualizations
        self.data_explorer.explore_data()

        # Generate and print insights report
        insights_report = self.data_explorer.generate_insights_report()
        print(insights_report)

    def analyze_correlations(self):
        """
        Analyze and visualize feature correlations.
        """
        corr_matrix = self.data.corr()
        self.visualizer.plot_correlation_matrix(self.data)

        # Print top correlations with price_range
        top_correlations = corr_matrix['price_range'].sort_values(ascending=False)
        print("Top correlations with price_range:")
        print(top_correlations)

    def analyze_feature_importance(self):
        """
        Analyze feature importance using a simple Random Forest model.
        """

        X = self.data.drop('price_range', axis=1)
        y = self.data['price_range']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X_train, y_train)

        feature_importance = rf.feature_importances_
        feature_names = X.columns

        self.visualizer.plot_feature_importance(feature_importance, feature_names)

    def run_eda(self):
        """
        Run the complete EDA process.
        """
        self.preprocess_data()
        self.perform_eda()
        self.analyze_correlations()
        self.analyze_feature_importance()

if __name__ == "__main__":
    eda = EDA(os.path.join("..", Config.TRAIN_DATA_PATH))
    eda.run_eda()