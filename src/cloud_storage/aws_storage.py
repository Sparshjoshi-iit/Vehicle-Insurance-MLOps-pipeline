import os
import shutil
from pathlib import Path

class LocalStorage:
    def __init__(self, base_model_dir: str = "saved_models"):
        self.base_model_dir = Path(base_model_dir)
        self.base_model_dir.mkdir(parents=True, exist_ok=True)

    def upload_file(self, source_path: str, model_name: str):
        """Simulate S3 upload by copying to local storage"""
        dest_path = self.base_model_dir / model_name
        shutil.copy(source_path, dest_path)

    def download_file(self, model_name: str, destination_path: str):
        """Simulate S3 download by copying from local storage"""
        source_path = self.base_model_dir / model_name
        shutil.copy(source_path, destination_path)

    def model_exists(self, model_name: str) -> bool:
        """Simulate S3 object existence check"""
        return (self.base_model_dir / model_name).exists()

    def get_model_path(self, model_name: str) -> str:
        """Return full local path"""
        return str(self.base_model_dir / model_name)
