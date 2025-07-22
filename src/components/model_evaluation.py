import sys
import numpy as np
from src.exception import MyException
from src.logger import logging
from src.entity.artifact_entity import DataTransformationArtifact, ModelTrainerArtifact, ModelEvaluationArtifact
from src.entity.config_entity import ModelEvaluationConfig
from src.utils.main_utils import load_object
from sklearn.metrics import r2_score, roc_auc_score
from src.entity.artifact_entity import RegressionMetricArtifact # Assuming this is in your artifacts

class ModelEvaluation:
    def __init__(self, model_eval_config: ModelEvaluationConfig,
                 data_transformation_artifact: DataTransformationArtifact,
                 model_trainer_artifact: ModelTrainerArtifact):
        try:
            self.model_eval_config = model_eval_config
            self.data_transformation_artifact = data_transformation_artifact
            self.model_trainer_artifact = model_trainer_artifact
        except Exception as e:
            raise MyException(e, sys) from e

    def get_production_models(self):
        """
        Loads the currently deployed production models.
        """
        try:
            production_models = {}
            model_paths = {
                "premium_model": self.model_eval_config.production_model_path1,
                "cost_model": self.model_eval_config.production_model_path2,
                "propensity_model": self.model_eval_config.production_model_path3,
                "churn_model": self.model_eval_config.production_model_path4,
            }
            for name, path in model_paths.items():
                # ### THE FIX IS HERE ###
                # We now catch `MyException` because the `load_object` utility
                # wraps the original FileNotFoundError.
                try:
                    production_models[name] = load_object(file_path=path)
                    logging.info(f"Loaded production model: {name} from {path}")
                except MyException as e:
                    # Check if the underlying error is a FileNotFoundError
                    if "No such file or directory" in str(e):
                        logging.warning(f"Production model not found for {name} at {path}. This is NORMAL on the first run.")
                        production_models[name] = None
                    else:
                        # If it's a different error, we should raise it
                        raise e
            return production_models
        except Exception as e:
            raise MyException(e, sys) from e

    def evaluate_models(self) -> ModelEvaluationArtifact:
        """
        Evaluates the newly trained models against the production models.
        """
        try:
            # Load all test data artifacts as objects and convert to CSR format
            logging.info("Loading test data for evaluation and converting to CSR format.")
            test_arr_premium = load_object(file_path=self.data_transformation_artifact.transformed_test_file_path).tocsr()
            test_arr_cost = load_object(file_path=self.data_transformation_artifact.transformed_test_file_path2).tocsr()
            test_arr_propensity = load_object(file_path=self.data_transformation_artifact.transformed_test_file_path3).tocsr()
            test_arr_churn = load_object(file_path=self.data_transformation_artifact.transformed_test_file_path4).tocsr()

            # Separate features and targets
            X_test = test_arr_premium[:, :-1]
            y_test_premium = test_arr_premium[:, -1].toarray().ravel()
            y_test_cost = test_arr_cost[:, -1].toarray().ravel()
            y_test_propensity = test_arr_propensity[:, -1].toarray().ravel()
            y_test_churn = test_arr_churn[:, -1].toarray().ravel()
            logging.info("Test features and targets separated.")

            # Load newly trained models from trainer artifact
            trained_premium_model = load_object(file_path=self.model_trainer_artifact.trained_model1_file_path)
            
            # Load production models
            production_models = self.get_production_models()

            # --- Evaluation Logic ---
            # We will compare the primary 'premium_model' R2 score.
            
            # Get score for the newly trained premium model
            trained_model_r2 = r2_score(y_test_premium, trained_premium_model.predict(X_test))
            logging.info(f"Newly trained premium model R2 score: {trained_model_r2}")

            # Get score for the production premium model
            production_premium_model = production_models.get("premium_model")
            production_model_r2 = -np.inf  # Default to a very low score
            if production_premium_model:
                production_model_r2 = r2_score(y_test_premium, production_premium_model.predict(X_test))
                logging.info(f"Production premium model R2 score: {production_model_r2}")

            # Compare scores to decide if the new model is accepted
            is_model_accepted = trained_model_r2 > production_model_r2
            difference = trained_model_r2 - production_model_r2
            logging.info(f"Model acceptance status: {is_model_accepted} (Difference: {difference})")
            
            # Create the final artifact
            model_evaluation_artifact = ModelEvaluationArtifact(
                is_model_accepted=is_model_accepted,
                changed_accuracy=difference,
                trained_model1_path=self.model_trainer_artifact.trained_model1_file_path,
                trained_model2_path=self.model_trainer_artifact.trained_model2_file_path,
                trained_model3_path=self.model_trainer_artifact.trained_model3_file_path,
                trained_model4_path=self.model_trainer_artifact.trained_model4_file_path,
            )
            return model_evaluation_artifact

        except Exception as e:
            raise MyException(e, sys) from e

    def initiate_model_evaluation(self) -> ModelEvaluationArtifact:
        try:
            logging.info("Starting model evaluation.")
            model_evaluation_artifact = self.evaluate_models()
            logging.info("Model evaluation completed.")
            return model_evaluation_artifact
        except Exception as e:
            raise MyException(e, sys) from e
