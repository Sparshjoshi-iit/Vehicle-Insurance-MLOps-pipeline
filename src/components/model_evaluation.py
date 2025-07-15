import sys
import numpy as np
import json
from pathlib import Path
from sklearn.metrics import f1_score, r2_score, mean_squared_error
from dataclasses import dataclass
from src.logger import logging
from src.utils.main_utils import load_numpy_array_data, load_object
from src.entity.s3_estimator import LocalProj1Estimator
from src.entity.config_entity import ModelEvaluationConfig
from src.entity.artifact_entity import (
    DataTransformationArtifact, ModelTrainerArtifact, ModelEvaluationArtifact
)
from src.exception import MyException


@dataclass
class EvaluateModelResponse:
    f1_model1: float
    r2_model2: float
    rmse_model2: float
    r2_model3: float
    rmse_model3: float
    is_model_accepted: bool
    difference: float


class ModelEvaluation:
    def __init__(self,
                 model_eval_config: ModelEvaluationConfig,
                 data_transformation_artifact: DataTransformationArtifact,
                 model_trainer_artifact: ModelTrainerArtifact):
        self.model_eval_config = model_eval_config
        self.data_transformation_artifact = data_transformation_artifact
        self.model_trainer_artifact = model_trainer_artifact

    def evaluate_model(self) -> EvaluateModelResponse:
        try:
            estimator = LocalProj1Estimator()
            logging.info("Model evaluation being started")
            # Load test arrays
            test_arr1 = load_numpy_array_data(self.data_transformation_artifact.transformed_test_file_path)
            test_arr2 = load_numpy_array_data(self.data_transformation_artifact.transformed_test_file_path2)

            x_test1, y_test1 = test_arr1[:, :-1], test_arr1[:, -1]
            x_test2, y_test2 = test_arr2[:, :-1], test_arr2[:, -1]
            logging.info("Loading models")
            # Load trained models
            model1 = load_object(self.model_trainer_artifact.trained_model1_file_path)
            model2 = load_object(self.model_trainer_artifact.trained_model2_file_path)
            model3 = load_object(self.model_trainer_artifact.trained_model3_file_path)
            logging.info('Starting classification')
            # Model 1: Classification
            y_test1_binary = np.where(y_test1 != 0, 1, 0)
            pred_model1 = model1.predict(x_test1)
            f1 = f1_score(y_test1_binary, pred_model1)

            logging.info('Starting Regression')
            # Model 2: Regression on filtered subset
            final_pred = np.zeros_like(y_test1_binary, dtype=float)

            # Filter indices
            non_zero_indices = np.where(pred_model1 != 0)[0]
            if len(non_zero_indices) > 0:
                x_test2_filtered = x_test2[non_zero_indices]
                y_test2_filtered = y_test1[non_zero_indices]
                
                pred_model2 = model2.predict(x_test2_filtered)

                # Assign predictions directly using fancy indexing
                final_pred[non_zero_indices] = pred_model2

                rmse2 = mean_squared_error(y_test2_filtered, pred_model2)
                r2_2 = r2_score(y_test2_filtered, pred_model2)
            else:
                rmse2 = -1
                r2_2 = -1
            logging.info('Started regression for "Premium" target variable')
            # Model 3: Full regression
            pred_model3 = model3.predict(x_test2)
            rmse3 = mean_squared_error(y_test2, pred_model3)
            r2_3 = r2_score(y_test2, pred_model3)

            logging.info(f"Model 1 F1: {f1:.4f}")
            logging.info(f"Model 2 R2: {r2_2:.4f}, RMSE: {rmse2:.4f}")
            logging.info(f"Model 3 R2: {r2_3:.4f}, RMSE: {rmse3:.4f}")
            logging.info('saving to artifac/evaluation.json')
            # Compare against previous best (r2 of model2)
            score_file = Path("artifact/evaluation.json")
            prev_r2_2 = 0
            if score_file.exists():
                with open(score_file, "r") as f:
                    scores = json.load(f)
                    prev_r2_2 = scores.get("model2", {}).get("r2_score", 0)

            is_model_accepted = r2_2 > (prev_r2_2 + self.model_eval_config.changed_threshold_score)
            difference = r2_2 - prev_r2_2
            logging.info('saving the models after pred')
            if is_model_accepted:
                estimator.save_model(model1, "model1")
                estimator.save_model(model2, "model2")
                estimator.save_model(model3, "model3")

                score_file.parent.mkdir(parents=True, exist_ok=True)
                with open(score_file, "w") as f:
                    json.dump({
                        "model1": {"f1_score": round(float(f1), 4)},
                        "model2": {"r2_score": round(r2_2, 4), "rmse": round(rmse2, 4)},
                        "model3": {"r2_score": round(r2_3, 4), "rmse": round(rmse3, 4)}
                    }, f, indent=4)

            return EvaluateModelResponse(
                f1_model1=f1,
                r2_model2=r2_2,
                rmse_model2=rmse2,
                r2_model3=r2_3,
                rmse_model3=rmse3,
                is_model_accepted=is_model_accepted,
                difference=difference
            )

        except Exception as e:
            raise MyException(e, sys)

    def initiate_model_evaluation(self) -> ModelEvaluationArtifact:
        try:
            logging.info("Starting Model Evaluation...")
            response = self.evaluate_model()
            logging.info('completed successfuly')
            return ModelEvaluationArtifact(
                is_model_accepted=response.is_model_accepted,
                changed_accuracy=response.difference,
                trained_model1_path=self.model_trainer_artifact.trained_model1_file_path,
                trained_model2_path=self.model_trainer_artifact.trained_model2_file_path,
                trained_model3_path=self.model_trainer_artifact.trained_model3_file_path,
            )
            
            
            
        except Exception as e:
            raise MyException(e, sys)
