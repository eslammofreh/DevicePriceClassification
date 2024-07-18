from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from utils.data_preprocessor import DataPreprocessor
import joblib
import os
from config import Config
from models.model_trainer import ModelTrainer

app = Flask(__name__)

class API:
    def __init__(self, model_path):
        self.model_path = model_path
        
    def preprocess_input(self, data):
        if os.path.exists(Config.PREPROCESSOR_SAVE_PATH): 
            self.preprocessor = DataPreprocessor.load_preprocessor(Config.PREPROCESSOR_SAVE_PATH)
        data = DataPreprocessor.add_engineered_features(data)
        data = DataPreprocessor.handle_infinite_values(data)
        data = self.preprocessor.transform(data)
        return data

    def predict(self, data):
        output = ''
        if os.path.exists(self.model_path):
            self.model = joblib.load(self.model_path)
            processed_data = self.preprocess_input(data)
            prediction = self.model.predict(processed_data)
            output = int(prediction[0])
        return jsonify(output) # Return a simple JSON with just the integer

    def run_prediction(self):
        data = input(Config.DEVICES_SPECS_MSG)
        data = [float(x.strip()) for x in data.split(',')]
        data = pd.DataFrame([data], columns=Config.COLUMNS)
        prediction = self.predict(data)
        print(f"{Config.PRED_PRICE_RANGE, prediction}")

    def run_server(self):
        @app.route(Config.API_ENDPOINT['PREDICT'], methods=['POST'])
        def predict():
            data = request.get_json(force=True)
            data = pd.DataFrame([data])
            prediction = self.predict(data)
            return prediction

        @app.route(Config.API_ENDPOINT['TRAIN'], methods=['GET'])
        def train():
            trainer = ModelTrainer(Config.TRAIN_DATA_PATH, Config.MODEL_SAVE_PATH)
            trainer.train()
            return jsonify(message=Config.MODEL_TRAINED_MSG)

        app.run(port=5000, debug=True)