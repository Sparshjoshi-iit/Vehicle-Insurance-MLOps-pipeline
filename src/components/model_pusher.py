import os
import sys
import shutil
from src.logger import logging
from src.exception import MyException
from src.entity.config_entity import ModelPusherConfig
from src.entity.artifact_entity import ModelTrainerArtifact, ModelEvaluationArtifact, ModelPusherArtifact

class ModelPusher:
    def __init__(
        self,
        model_pusher_config: ModelPusherConfig,
        model_trainer_artifact: ModelTrainerArtifact,
        model_evaluation_artifact: ModelEvaluationArtifact,
    ):
        try:
            logging.info("Entered ModelPusher constructor ")
            self.model_pusher_config = model_pusher_config
            self.model_trainer_artifact = model_trainer_artifact
            self.model_evaluation_artifact = model_evaluation_artifact
        except Exception as e:
            raise MyException(e, sys)

    def initiate_model_pusher(self) -> ModelPusherArtifact:
        try:
            logging.info("Starting model push process...")

            # Ensure destination folder exists
            os.makedirs(self.model_pusher_config.saved_model_dir, exist_ok=True)

            # Prepare all model file paths
            model_save_map = {
                "model1.pkl": self.model_trainer_artifact.trained_model1_file_path,
                "model2.pkl": self.model_trainer_artifact.trained_model2_file_path,
                "model3.pkl": self.model_trainer_artifact.trained_model3_file_path,
            }

            pushed_model_paths = {}

            for filename, source_path in model_save_map.items():
                dest_path = os.path.join(self.model_pusher_config.saved_model_dir, filename)
                shutil.copy2(src=source_path, dst=dest_path)
                pushed_model_paths[filename] = dest_path
                logging.info(f"Pushed {filename} to {dest_path}")

            # Build and return artifact
            model_pusher_artifact = ModelPusherArtifact(
                saved_model_dir=self.model_pusher_config.saved_model_dir,
                model1_path=pushed_model_paths["model1.pkl"],
                model2_path=pushed_model_paths["model2.pkl"],
                model3_path=pushed_model_paths["model3.pkl"]
            )

            logging.info(f"ModelPusherArtifact created: {model_pusher_artifact}")
            return model_pusher_artifact

        except Exception as e:
            raise MyException(e, sys)
