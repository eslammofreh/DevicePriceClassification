from utils.data_loader import DataLoader
from utils.data_preprocessor import DataPreprocessor
from models.random_forest_model import RandomForestModel
from models.stacking_model import StackingModel
from utils.visualization import Visualizer
from models.model_evaluation import ModelEvaluator
import joblib
from config import Config
from utils.model_utils import ModelUtils

class ModelTrainer:
    def __init__(self, data_path, model_save_path):
        self.data_path = data_path
        self.model_save_path = model_save_path
        self.data_loader = DataLoader(data_path)
        self.preprocessor = DataPreprocessor()
        self.visualizer = Visualizer()

    def train(self):
        # Load and preprocess data
        data_loader = DataLoader(self.data_path)
        data = data_loader.load_data()

        preprocessor = DataPreprocessor()
        data = preprocessor.handle_missing_values(data)
        data = preprocessor.handle_infinite_values(data)
        data = preprocessor.add_engineered_features(data) # Create new features to enhance the model performance.

        X_train, X_test, y_train, y_test = preprocessor.split_data(data)

        X_train = preprocessor.handle_infinite_values(X_train)
        X_test = preprocessor.handle_infinite_values(X_test)

        preprocessing_pipeline = self.preprocessor.create_preprocessor()
        X_train_processed = preprocessing_pipeline.fit_transform(X_train)
        X_test_processed = preprocessing_pipeline.transform(X_test)
        
        self.preprocessor.save_preprocessor(preprocessing_pipeline, Config.PREPROCESSOR_SAVE_PATH)

        # Train and evaluate Random Forest model
        rf_model = RandomForestModel()
        rf_model.train(X_train_processed, y_train)
        rf_evaluation = ModelUtils.evaluate_model(rf_model, X_test_processed, y_test, Config.MODELS['RANDOM_FOREST'])
        ModelUtils.visualize_model(rf_model, X_test_processed, y_test, X_test.columns.tolist(), self.visualizer)
        ModelUtils.interpret_model(rf_model, X_test_processed)

        # Train and evaluate Stacking model
        stacking_model = StackingModel()
        stacking_model.train(X_train_processed, y_train)
        stacking_evaluation = ModelUtils.evaluate_model(stacking_model, X_test_processed, y_test, Config.MODELS['STACKING'])
        ModelUtils.visualize_model(stacking_model, X_test_processed, y_test, X_test.columns.tolist(), self.visualizer)

        best_model = ModelUtils.compare_models(rf_evaluation, stacking_evaluation, Config.EVAL_TECHS['ACCURACY'])

        # Save the best model (assuming Random Forest is better)
        if best_model == Config.MODELS['RANDOM_FOREST']:
           joblib.dump(rf_model.model, self.model_save_path)
        else:
           joblib.dump(stacking_model.model, self.model_save_path)
        