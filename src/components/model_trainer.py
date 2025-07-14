import sys
from typing import Tuple
from src.logger import logging
import numpy as np
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestClassifier
from src.entity.artifact_entity import DataTransformationArtifact, ModelTrainerArtifact, ClassificationMetricArtifact, RegressionMetricArtifact
from src.entity.config_entity import ModelTrainerConfig
from src.exception import MyException
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.metrics import root_mean_squared_error, r2_score, accuracy_score, f1_score, precision_score, recall_score
from src.utils.main_utils import load_numpy_array_data, load_object, save_object
from src.entity.estimator import MyModel

# creating the model_trainer for the target variables
class ModelTrainer:
    def __init__(self, data_transformation_artifact: DataTransformationArtifact, model_trainer_config: ModelTrainerConfig):
        self.data_transformation_artifact = data_transformation_artifact
        self.model_trainer_config = model_trainer_config

    def get_model1_object_and_report(self, train: np.array, test: np.array) -> Tuple[object, ClassificationMetricArtifact]:
        try:
            x_train, y_train, x_test, y_test = train[:, :-1], train[:, -1], test[:, :-1], test[:, -1]
            y_train_filtered = np.where(y_train != 0, 1, 0)
            y_test_filtered = np.where(y_test != 0, 1, 0)
            logging.info('Training RandomForestClassifier for zeros')
            
            pipeline = Pipeline([
                ('smote', SMOTE(random_state=42)),
                ('clf', RandomForestClassifier(random_state=42, max_depth=4, min_samples_leaf=2, min_samples_split=3, criterion='log_loss'))
            ])
            pipeline.fit(x_train, y_train_filtered)
            model1 = pipeline
            logging.info('model1 trained successfuly')
            y_pred = model1.predict(x_test)
            accuracy = accuracy_score(y_test_filtered, y_pred)
            f1 = f1_score(y_test_filtered, y_pred)
            precision = precision_score(y_test_filtered, y_pred)
            recall = recall_score(y_test_filtered, y_pred)

            metric_artifact = ClassificationMetricArtifact(f1_score=f1, precision_score=precision, recall_score=recall)
            return model1, metric_artifact

        except Exception as e:
            raise MyException(e, sys) from e

    def get_model2_object_and_report(self, train: np.array, test: np.array, pred_of_model1: np.array,pred_of_model1_test:np.array) -> Tuple[object,RegressionMetricArtifact]:
        try:
            x_train, y_train, x_test, y_test = train[:, :-1], train[:, -1], test[:, :-1], test[:, -1]
            mask1=(pred_of_model1!=0).astype(int)
            mask2=(pred_of_model1_test!=0).astype(int)
            logging.info('creating modified data for non-zero values')
            x_train_modified = x_train[mask1]
            y_train_modified = y_train[mask1]
            x_test_modified = x_test[mask2]
            y_test_modified = y_test[mask2]
            
            xgb = XGBRegressor(
                subsample=0.8, reg_lambda=5, reg_alpha=1, max_depth=3,
                learning_rate=0.01, colsample_bytree=1.0, n_estimators=100
            )
            xgb.fit(x_train_modified, y_train_modified)
            model2 = xgb
            logging.info('model2 trained successfuly')
            y_pred = model2.predict(x_test_modified)
            rmse = root_mean_squared_error(y_test_modified, y_pred)
            r2 = r2_score(y_test_modified, y_pred)
            metric_artifact = RegressionMetricArtifact(rmse=rmse, r2_score=r2)
            return model2, metric_artifact

        except Exception as e:
            raise MyException(e, sys) from e

    def get_model3_object_and_report(self, train: np.array, test: np.array) -> Tuple[object, RegressionMetricArtifact]:
        try:
            # Assuming last column is the 3rd target variable
            x_train, y_train, x_test, y_test = train[:, :-1], train[:, -1], test[:, :-1], test[:, -1]
            logging.info('training model 3 for "Premium" target feature')
            xgb = XGBRegressor(
                n_estimators=1160,
                max_depth=10,
                learning_rate=0.033275614993222624,
                subsample=0.9759172043150037,
                colsample_bytree=0.8239006311477104,
                reg_alpha=1.2920225771570368,
                reg_lambda=0.13266023283593514,
                min_child_weight=3,
                random_state=42
            )
            xgb.fit(x_train, y_train)
            model3 = xgb
            logging.info("model3 trained successfuly")
            y_pred = model3.predict(x_test)
            rmse = root_mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            metric_artifact = RegressionMetricArtifact(rmse=rmse, r2_score=r2)
            return model3, metric_artifact

        except Exception as e:
            raise MyException(e, sys) from e

    def initiate_all_models_trainer(self) -> ModelTrainerArtifact:
        logging.info("Entered initiate_model_trainer method of ModelTrainer class")
        try:
            train_arr = load_numpy_array_data(file_path=self.data_transformation_artifact.transformed_train_file_path)
            test_arr = load_numpy_array_data(file_path=self.data_transformation_artifact.transformed_test_file_path)
            
            train_arr2 = load_numpy_array_data(file_path=self.data_transformation_artifact.transformed_train_file_path2)
            test_arr2 = load_numpy_array_data(file_path=self.data_transformation_artifact.transformed_test_file_path2)
            
            trained_model1, metric1_artifact = self.get_model1_object_and_report(train=train_arr, test=test_arr)
            preprocessing_obj = load_object(file_path=self.data_transformation_artifact.transformed_object_file_path)
            my_model1 = MyModel(preprocessing_object=preprocessing_obj, trained_model_object=trained_model1)
            save_object(self.model_trainer_config.trained_model1_file_path, my_model1)

            x_train = train_arr[:, :-1]
            x_test= train_arr[:,:-1]
            pred_model1_train = trained_model1.predict(x_train)
            pred_model1_test=trained_model1.predict(x_test)

            trained_model2, metric2_artifact = self.get_model2_object_and_report(train=train_arr2, test=test_arr2, pred_of_model1=pred_model1_train,pred_of_model1_test=pred_model1_test)
            my_model2 = MyModel(preprocessing_object=preprocessing_obj, trained_model_object=trained_model2)
            save_object(self.model_trainer_config.trained_model2_file_path, my_model2)

            trained_model3, metric3_artifact = self.get_model3_object_and_report(train=train_arr2, test=test_arr2)
            my_model3 = MyModel(preprocessing_object=preprocessing_obj, trained_model_object=trained_model3)
            save_object(self.model_trainer_config.trained_model3_file_path, my_model3)
            logging.info('saved the object')
            model_trainer_artifact = ModelTrainerArtifact(
                trained_model1_file_path=self.model_trainer_config.trained_model1_file_path,
                trained_model2_file_path=self.model_trainer_config.trained_model2_file_path,
                trained_model3_file_path=self.model_trainer_config.trained_model3_file_path,
                metric1_artifact=metric1_artifact,
                metric2_artifact=metric2_artifact,
                metric3_artifact=metric3_artifact
            )
            return model_trainer_artifact

        except Exception as e:
            raise MyException(e, sys) from e