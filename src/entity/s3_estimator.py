import os
import joblib
from pathlib import Path
from src.exception import MyException
import sys


class LocalProj1Estimator:
    def __init__(self, model_dir: str = "saved_models"):
        try:
            self.model_dir = Path(model_dir)
            self.model_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            raise MyException(e, sys) from e

    def save_model(self, model, model_name: str) -> None:
        """
        Saves the model locally in the specified directory.
        """
        try:
            model_path = self.model_dir / f"{model_name}.pkl"
            joblib.dump(model, model_path)
        except Exception as e:
            raise MyException(e, sys) from e

    def get_model_path(self, model_name: str) -> str:
        """
        Returns the model path if it exists, else None.
        """
        try:
            model_path = self.model_dir / f"{model_name}.pkl"
            return str(model_path) if model_path.exists() else None
        except Exception as e:
            raise MyException(e, sys) from e

    def load_model(self, model_name: str):
        """
        Loads and returns the model if it exists.
        """
        try:
            model_path = self.get_model_path(model_name)
            if model_path:
                return joblib.load(model_path)
            else:
                return None
        except Exception as e:
            raise MyException(e, sys) from e

    def predict(self, model_name: str, x):
        """
        Loads the model and performs prediction.
        """
        try:
            model = self.load_model(model_name)
            if model is None:
                raise FileNotFoundError(f"Model '{model_name}' not found in {self.model_dir}")
            return model.predict(x)
        except Exception as e:
            raise MyException(e, sys) from e
