class Config:

    # Model Parameters
    RANDOM_FOREST_PARAMS = {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2']
    }

    STACKING_MODEL_PARAMS = {
        'rf__n_estimators': [100, 200],
        'rf__max_depth': [None, 10, 20],
        'gb__n_estimators': [100, 200],
        'gb__max_depth': [3, 5],
        'final_estimator__C': [0.1, 1, 10]
    }

    CROSS_VALIDATION = 5

    # Data Parameters
    COLUMNS = ['battery_power', 'blue', 'clock_speed', 'dual_sim', 'fc', 'four_g',
                   'int_memory', 'm_dep', 'mobile_wt', 'n_cores', 'pc', 'px_height',
                   'px_width', 'ram', 'sc_h', 'sc_w', 'talk_time', 'three_g',
                   'touch_screen', 'wifi']
    NUMERIC_FEATURES = ['battery_power', 'clock_speed', 'fc', 'int_memory', 'm_dep', 'mobile_wt',
                            'n_cores', 'pc', 'px_height', 'px_width', 'ram', 'sc_h', 'sc_w', 'talk_time', 
                           'screen_area', 'pixel_density', 'battery_efficiency', 'speed_per_core', 'memory_per_core',
           'total_camera_mp', 'ram_battery_interaction']
        
    BINARY_FEATURES = ['blue', 'dual_sim', 'four_g', 'three_g', 'touch_screen', 'wifi']

    # Model Selection
    MODELS = {'RANDOM_FOREST':'Random Forest', 'STACKING':'Stacking'}
    OUTLIER_TECHS = {'IQR':'IQR', 'ZSCORE':'zscore'}
    ACTIONS = {'TRAIN':'train', 'PREDICT':'predict', 'RUN_SERVER':'run_server'}
    CLASSES = {'LOW':'low', 'MEDIUM':'medium', 'HIGH':'high', 'VERY_HIGH':'very_high'}
    EVAL_TECHS = {'ACCURACY':'accuracy', 'PRECISION':'precision', 'RECALL':'recall', 'F1_SCORE':'f1_score', 'CLASSIFICATION_REPORT':'classification_report'}
    API_ENDPOINT = {'TRAIN':'/train', 'PREDICT':'/predict'}

    # File Paths
    MODEL_SAVE_PATH = 'results/models/best_model.pkl'
    PREPROCESSOR_SAVE_PATH = 'results/models/preprocessor.pkl'
    VISUALIZATION_OUTPUT_DIR = 'results/data_exploration_results'
    TRAIN_DATA_PATH = 'data/train.csv'
    TEST_DATA_PATH = 'data/test.csv'

    # Messages displayed on the screen
    PROJECT_NAME = "Device Price Classification System"
    ACTION_MSG = 'Action to perform'
    DEVICES_SPECS_MSG = "Enter device specifications (comma-separated values): "
    PRED_PRICE_RANGE = 'Predicted price range:'
    EVALUATION_MSG = " Model Evaluation:"
    CLASS_REPORT_MSG = "\nClassification Report:"
    MODEL_TRAINED_MSG = "Model trained successfully"
    BEST_PARAMS_MSG = "Best parameters:"
    DATASET_INFO_MSG = "Dataset Information:"
    SUMMARY_STAT_MSG = "\nSummary Statistics:"
    MISS_VALUES_MSG = "\nMissing Values:"
    MISSMATCH_FEATURE_IMPORT_NAMES_MSG = "The length of feature importances does not match the length of feature names."
    ENG_FEATURES_ADDED_MSG = "Engineered features added:"

    # Visualization Titles and Labels
    TARGET_DISTRIBUTION_TITLE = 'Distribution of Price Ranges'
    TARGET_X_LABEL = 'Price Range'
    TARGET_Y_COUNT_LABEL = 'Count'
    TARGET_DISTRIBUTION_FILENAME = 'target_distribution.png'
    
    FEATURE_DISTRIBUTIONS_FILENAME = 'feature_distributions.png'
    CORRELATION_MATRIX_TITLE = 'Correlation Matrix of Features'
    CORRELATION_MATRIX_FILENAME = 'correlation_matrix.png'
    
    PAIRPLOT_TITLE = 'Pairplot of Key Features'
    PAIRPLOT_FILENAME = 'pairplot.png'
    
    FEATURE_IMPORTANCE_TITLE = 'Feature Importance'
    FEATURE_IMPORTANCE_FILENAME = 'feature_importance.png'
    
    CONFUSION_MATRIX_TITLE = 'Confusion Matrix'
    CONFUSION_MATRIX_YLABEL = 'Actual'
    CONFUSION_MATRIX_XLABEL = 'Predicted'
    CONFUSION_MATRIX_FILENAME = 'confusion_matrix.png'

    # DataExplorer specific
    PAIRPLOT_FEATURES = ['battery_power', 'ram', 'px_height', 'px_width', 'mobile_wt', 'price_range']
    INSIGHTS_REPORT_TITLE = "Data Exploration Insights:"
    TARGET_DISTRIBUTION_INSIGHT_TITLE = "Target Distribution:"
    DATASET_CONTAINS_MSG = "The dataset contains"
    MOBILE_DEVICES_MSG = "mobile devices."
    PRICE_RANGE_MSG = "Price range"
    DEVICES_MSG = "devices"
    HIGH_CORR_FEATURES_TITLE = "Highly Correlated Features with Price Range:"
    CORRELATION_MSG = "correlation of"
    FEATURE_DISTRIBUTION_LIST = ['ram', 'battery_power', 'px_height', 'px_width']
    DISTRIBUTION_TITLE = "Distribution"
    AVERAGE_MSG = "Average"
    FOR_PRICE_RANGE_MSG = "for price range"
    INSIGHTS_REPORT_FILENAME = 'insights_report.txt'
