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
            logging.info("Entered ModelPusher constructor")
            self.model_pusher_config = model_pusher_config
            self.model_trainer_artifact = model_trainer_artifact
            self.model_evaluation_artifact = model_evaluation_artifact
        except Exception as e:
            raise MyException(e, sys) from e

    def initiate_model_pusher(self) -> ModelPusherArtifact:
        try:
            logging.info("Starting model push process...")

            # Step 1: Check if the new models were accepted
            if not self.model_evaluation_artifact.is_model_accepted:
                logging.info("New models were not accepted by the evaluation step. Skipping model pusher.")
                # ### MODIFIED ###: Returning None as the artifact structure doesn't support a "not pushed" state.
                # The pipeline logic should handle a None return.
                return None

            logging.info("New models were accepted. Proceeding to push models to production.")

            # Step 2: Define source and destination paths
            trained_model_paths = {
                "premium_model": self.model_trainer_artifact.trained_model1_file_path,
                "cost_model": self.model_trainer_artifact.trained_model2_file_path,
                "propensity_model": self.model_trainer_artifact.trained_model3_file_path,
                "churn_model": self.model_trainer_artifact.trained_model4_file_path,
            }

            production_model_paths = {
                "premium_model": self.model_pusher_config.production_model_path1,
                "cost_model": self.model_pusher_config.production_model_path2,
                "propensity_model": self.model_pusher_config.production_model_path3,
                "churn_model": self.model_pusher_config.production_model_path4,
            }
            
            logging.info("Copying accepted models to production directory.")
            for model_name in trained_model_paths:
                source_path = trained_model_paths[model_name]
                dest_path = production_model_paths[model_name]

                # Create the destination directory (e.g., 'saved_models/premium_model/')
                os.makedirs(os.path.dirname(dest_path), exist_ok=True)
                
                # Copy the model file (e.g., to 'saved_models/premium_model/model.pkl')
                shutil.copy(source_path, dest_path)
                logging.info(f"Pushed {model_name} from {source_path} to {dest_path}")

            # ### MODIFIED ### - Step 3: Build and return the final artifact as per the provided definition
            model_pusher_artifact = ModelPusherArtifact(
                saved_model_dir=self.model_pusher_config.saved_model_dir,
                model1_path=production_model_paths["premium_model"],
                model2_path=production_model_paths["cost_model"],
                model3_path=production_model_paths["propensity_model"],
                model4_path=production_model_paths["churn_model"]
            )

            logging.info(f"ModelPusherArtifact created: {model_pusher_artifact}")
            return model_pusher_artifact

        except Exception as e:
            raise MyException(e, sys) from e
