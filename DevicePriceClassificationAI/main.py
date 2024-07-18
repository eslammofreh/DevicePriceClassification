import argparse
from utils.data_loader import DataLoader
from utils.data_preprocessor import DataPreprocessor
from models.model_trainer import ModelTrainer
from api.api import API
from config import Config

def main():
    parser = argparse.ArgumentParser(description=Config.PROJECT_NAME)
    parser.add_argument('action', choices=Config.ACTIONS.values(), help=Config.ACTION_MSG)
    args = parser.parse_args()

    if args.action.lower() == Config.ACTIONS['TRAIN']:
        trainer = ModelTrainer(Config.DATA_PATH, Config.MODEL_SAVE_PATH)
        trainer.train()
    elif args.action.lower() == Config.ACTIONS['PREDICT']:
        api = API(Config.MODEL_SAVE_PATH)
        api.run_prediction()
    elif args.action.lower() == Config.ACTIONS['RUN_SERVER']:
        api = API(Config.MODEL_SAVE_PATH)
        api.run_server()

if __name__ == "__main__":
    main()