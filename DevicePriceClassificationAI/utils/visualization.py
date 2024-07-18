import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os
from config import Config

class Visualizer:
    def __init__(self, output_dir=None):
        """
        Initialize the Visualizer class.

        Args:
            output_dir (str): Directory to save results.
        """
        if output_dir is not None:
            self.output_dir = output_dir
        else:
            self.output_dir = Config.VISUALIZATION_OUTPUT_DIR
        os.makedirs(self.output_dir, exist_ok=True)

    def save_plot(self, fig, filename):
        """
        Save the given figure to the output directory.

        Args:
            fig (matplotlib.figure.Figure): Figure to save.
            filename (str): Name of the file to save the figure as.
        """
        fig.savefig(os.path.join(self.output_dir, filename))
        plt.close(fig)

    def plot_target_distribution(self, data):
        """
        Plot the distribution of the target variable (price_range).

        Args:
            data (pd.DataFrame): The dataset containing the target variable.
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.countplot(x='price_range', data=data, ax=ax)
        ax.set_title(Config.TARGET_DISTRIBUTION_TITLE)
        ax.set_xlabel(Config.TARGET_X_LABEL)
        ax.set_ylabel(Config.TARGET_Y_COUNT_LABEL)
        self.save_plot(fig, Config.TARGET_DISTRIBUTION_FILENAME)

    def plot_feature_distributions(self, data):
        """
        Plot the distribution of numeric features, grouped by price range.

        Args:
            data (pd.DataFrame): The dataset containing the features.
        """
        numeric_features = data.select_dtypes(include=[np.number]).columns
        fig, axes = plt.subplots(nrows=(len(numeric_features) + 1) // 3, ncols=3, figsize=(20, 5 * ((len(numeric_features) + 1) // 3)))
        for i, feature in enumerate(numeric_features):
            if feature != 'price_range':
                sns.histplot(data=data, x=feature, hue='price_range', multiple="stack", ax=axes[i // 3, i % 3])
        fig.tight_layout()
        self.save_plot(fig, Config.FEATURE_DISTRIBUTIONS_FILENAME)

    def plot_correlation_matrix(self, data):
        """
        Plot the correlation matrix of the features.

        Args:
            data (pd.DataFrame): The dataset containing the features.
        """
        corr_matrix = data.corr()
        fig, ax = plt.subplots(figsize=(15, 12))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', ax=ax)
        ax.set_title(Config.CORRELATION_MATRIX_TITLE)
        self.save_plot(fig, Config.CORRELATION_MATRIX_FILENAME)

    def plot_pairplot(self, data, features):
        """
        Create a pairplot for the specified features.

        Args:
            data (pd.DataFrame): The dataset containing the features.
            features (list): List of features to include in the pairplot.
        """
        fig = sns.pairplot(data[features], hue='price_range')
        fig.fig.suptitle(Config.PAIRPLOT_TITLE, y=1.02)
        self.save_plot(fig.fig,Config.PAIRPLOT_FILENAME)

    def plot_feature_importance(self, feature_importance, feature_names):
        """
        Plot the feature importance.

        Args:
            feature_importance (np.array): Array of feature importance values.
            feature_names (list): List of feature names.
        """
        # Verify lengths
        if len(feature_importance) != len(feature_names):
            raise ValueError(Config.MISSMATCH_FEATURE_IMPORT_NAMES_MSG)
        
        fig, ax = plt.subplots(figsize=(12, 8))
        feature_importance_df = pd.DataFrame({'feature': feature_names, 'importance': feature_importance})
        feature_importance_df = feature_importance_df.sort_values('importance', ascending=False)
        sns.barplot(x='importance', y='feature', data=feature_importance_df, ax=ax)
        ax.set_title(Config.FEATURE_IMPORTANCE_TITLE)
        self.save_plot(fig, Config.FEATURE_IMPORTANCE_FILENAME)
    
    def plot_confusion_matrix(self, y_true, y_pred, class_names=None):
        """
        Plot the confusion matrix.

        Args:
            y_true (array-like): True labels.
            y_pred (array-like): Predicted labels.
            class_names (list): List of class names (optional).
        """
        cm = confusion_matrix(y_true, y_pred)
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_ylabel(Config.CONFUSION_MATRIX_YLABEL)
        ax.set_xlabel(Config.CONFUSION_MATRIX_XLABEL)
        if class_names:
            ax.set_xticklabels(class_names)
            ax.set_yticklabels(class_names)
        ax.set_title(Config.CONFUSION_MATRIX_TITLE)
        self.save_plot(fig, Config.CONFUSION_MATRIX_FILENAME)

class DataExplorer:
    def __init__(self, data, visualizer=Config.VISUALIZATION_OUTPUT_DIR):
        """
        Initialize the DataExplorer class.

        Args:
            data (pd.DataFrame): The dataset to explore.
        """
        self.data = data
        self.visualizer = Visualizer(output_dir=visualizer)

    def explore_data(self):
        """
        Perform comprehensive data exploration and generate visualizations.
        """
        self.visualizer.plot_target_distribution(self.data)
        self.visualizer.plot_feature_distributions(self.data)
        self.visualizer.plot_correlation_matrix(self.data)
        self.visualizer.plot_pairplot(self.data, Config.PAIRPLOT_FEATURES)

    def generate_insights_report(self):
        """
        Generate a report with key insights from the data exploration.

        Returns:
            str: A string containing the insights report.
        """
        report = Config.INSIGHTS_REPORT_TITLE + "\n\n"

        # Target distribution insight
        price_range_counts = self.data['price_range'].value_counts()
        report += f"1. {Config.TARGET_DISTRIBUTION_INSIGHT_TITLE}\n"
        report += f"   - {Config.DATASET_CONTAINS_MSG} {len(self.data)} {Config.MOBILE_DEVICES_MSG}\n"
        for price_range, count in price_range_counts.items():
            report += f"   - {Config.PRICE_RANGE_MSG} {price_range}: {count} {Config.DEVICES_MSG} ({count / len(self.data) * 100:.2f}%)\n"
        report += "\n"

        # Correlation insights
        corr_matrix = self.data.corr()
        high_corr_features = corr_matrix[abs(corr_matrix['price_range']) > 0.5]['price_range'].index.tolist()
        report += f"2. {Config.HIGH_CORR_FEATURES_TITLE}\n"
        for feature in high_corr_features:
            if feature != 'price_range':
                corr = corr_matrix.loc['price_range', feature]
                report += f"   - {feature}: {Config.CORRELATION_MSG} {corr:.2f}\n"
        report += "\n"

        # Feature distribution insights
        for feature in Config.FEATURE_DISTRIBUTION_LIST:
            mean_by_price = self.data.groupby('price_range')[feature].mean()
            report += f"3. {feature.capitalize()} {Config.DISTRIBUTION_TITLE}:\n"
            for price_range, mean_value in mean_by_price.items():
                report += f"   - {Config.AVERAGE_MSG} {feature} {Config.FOR_PRICE_RANGE_MSG} {price_range}: {mean_value:.2f}\n"
            report += "\n"

        # Save the report
        with open(os.path.join(self.visualizer.output_dir, Config.INSIGHTS_REPORT_FILENAME), 'w') as f:
            f.write(report)

        return report
