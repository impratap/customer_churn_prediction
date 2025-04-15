import joblib
from src.exception import CustomException
from src.logger import logging
import sys

def load_model(path):
    try:
        logging.info("loading Saved Models")
        return joblib.load(path)
    except Exception as e:
        raise CustomException(e,sys)