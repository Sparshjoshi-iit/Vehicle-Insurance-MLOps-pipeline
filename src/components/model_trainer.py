import sys
import numpy as np
import pandas as pd
import lightgbm as lgb
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, r2_score, root_mean_squared_error
from src.entity.artifact_entity import (
    DataTransformationArtifact, 
    ModelTrainerArtifact, 
    ClassificationMetricArtifact, 
    RegressionMetricArtifact
)
from src.entity.config_entity import ModelTrainerConfig
from src.exception import MyException
from src.logger import logging
from src.utils.main_utils import load_numpy_array_data, load_object, save_object
from src.entity.estimator import MyModel

class ModelTrainer:
    def __init__(self, data_transformation_artifact: DataTransformationArtifact, model_trainer_config: ModelTrainerConfig):
        self.data_transformation_artifact = data_transformation_artifact
        self.model_trainer_config = model_trainer_config

    def _train_premium_regressor(self, X_train: np.array, y_train: np.array) -> XGBRegressor:
        """
        Trains an XGBoost regressor to predict the Premium.
        Uses pre-defined hyperparameters for speed.
        """
        logging.info("--- Training Premium Regressor ---")
        try:
            xgb = XGBRegressor(
                n_estimators=1200, learning_rate=0.011407321239447075, max_depth=9,
                subsample= 0.8574089820235964, colsample_bytree=0.7602796084440845, gamma=0.01096388272097397,
                reg_lambda=8.913626004073375, reg_alpha=2.2442669946137332, random_state=42, n_jobs=-1
            )
            xgb.fit(X_train, y_train)
            logging.info("Premium Regressor training complete.")
            return xgb
        except Exception as e:
            raise MyException(e, sys)
        

    def _train_claim_cost_regressor(self, X_train: np.array, y_train: np.array) -> lgb.LGBMRegressor:
        """
        Trains a LightGBM regressor to predict the Claim Cost (Severity).
        This model is trained ONLY on data where a claim occurred.
        """
        logging.info("--- Training Claim Cost (Severity) Regressor ---")
        try:
            # Filter data to include only instances where a claim was made (cost > 0)
            claim_indices = np.where(y_train > 0)[0]
            X_train_claims = X_train[claim_indices]
            y_train_claims = y_train[claim_indices]

            if len(y_train_claims) == 0:
                logging.warning("No claim data available for severity model training. Returning None.")
                return None
            
            # Log-transform the target for better performance with skewed cost data
            y_train_log = np.log1p(y_train_claims)

            lgbm = lgb.LGBMRegressor(
                objective='regression_l1', metric='mae', random_state=42, n_jobs=-1,
                learning_rate=0.05, num_leaves=30, max_depth=5,colsample_bytree=0.7,subsample=0.7
            )
            lgbm.fit(X_train_claims, y_train_log)
            logging.info("Claim Cost Regressor training complete.")
            return lgbm
        except Exception as e:
            raise MyException(e, sys)

    def _train_claim_propensity_classifier(self, X_train: np.array, y_train: np.array) -> lgb.LGBMClassifier:
        """
        Trains a LightGBM classifier to predict Claim Propensity (likelihood of a claim).
        """
        logging.info("--- Training Claim Propensity Classifier ---")
        try:
            lgbm_clf = lgb.LGBMClassifier(
                n_estimators=1300,learning_rate=0.15063954632929094, num_leaves= 76, max_depth=7, subsample=0.7732263127935013, colsample_bytree=0.6484832666001777,
                objective='binary', metric='auc', is_unbalance=True, random_state=42, n_jobs=-1
            )
            lgbm_clf.fit(X_train, y_train)
            logging.info("Claim Propensity Classifier training complete.")
            return lgbm_clf
        except Exception as e:
            raise MyException(e, sys)
        

    def _train_customer_churn_classifier(self, X_train: np.array, y_train: np.array) -> RandomForestClassifier:
        """
        Trains a RandomForest classifier to predict Customer Churn.
        """
        logging.info("--- Training Customer Churn Classifier ---")
        try:
            rf_clf = RandomForestClassifier(
                n_estimators=600, max_depth=24, min_samples_leaf=2,min_samples_split=2,
                class_weight='balanced', random_state=42, n_jobs=-1
            )
            rf_clf.fit(X_train, y_train)
            logging.info("Customer Churn Classifier training complete.")
            return rf_clf
        except Exception as e:
            raise MyException(e, sys)

    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        """
        Orchestrates the training of all four models, evaluates them, and saves the artifacts.
        """
        logging.info("Entered initiate_model_trainer method of ModelTrainer class")
        try:
            # --- Load Data ---
            logging.info("Loading sparse transformed data.")
            train_arr_premium = load_object(file_path=self.data_transformation_artifact.transformed_train_file_path).tocsr()
            test_arr_premium = load_object(file_path=self.data_transformation_artifact.transformed_test_file_path).tocsr()

            train_arr_cost = load_object(file_path=self.data_transformation_artifact.transformed_train_file_path2).tocsr()
            test_arr_cost = load_object(file_path=self.data_transformation_artifact.transformed_test_file_path2).tocsr()

            train_arr_propensity = load_object(file_path=self.data_transformation_artifact.transformed_train_file_path3).tocsr()
            test_arr_propensity = load_object(file_path=self.data_transformation_artifact.transformed_test_file_path3).tocsr()
            
            train_arr_churn = load_object(file_path=self.data_transformation_artifact.transformed_train_file_path4).tocsr()
            test_arr_churn = load_object(file_path=self.data_transformation_artifact.transformed_test_file_path4).tocsr()
            logging.info("Sparse data loaded and converted to CSR format.")

            # --- Separate Features (Sparse) and Targets (Dense) ---
            X_train = train_arr_premium[:, :-1]
            X_test = test_arr_premium[:, :-1]
            
            y_train_premium = train_arr_premium[:, -1].toarray().ravel()
            y_test_premium = test_arr_premium[:, -1].toarray().ravel()

            y_train_cost = train_arr_cost[:, -1].toarray().ravel()
            y_test_cost = test_arr_cost[:, -1].toarray().ravel()

            y_train_propensity = train_arr_propensity[:, -1].toarray().ravel()
            y_test_propensity = test_arr_propensity[:, -1].toarray().ravel()

            y_train_churn = train_arr_churn[:, -1].toarray().ravel()
            y_test_churn = test_arr_churn[:, -1].toarray().ravel()
            logging.info("Features and targets separated successfully.")
            
            # --- Train All Models ---
            premium_model = self._train_premium_regressor(X_train=X_train, y_train=y_train_premium)
            cost_model = self._train_claim_cost_regressor(X_train=X_train, y_train=y_train_cost)
            propensity_model = self._train_claim_propensity_classifier(X_train=X_train, y_train=y_train_propensity)
            churn_model = self._train_customer_churn_classifier(X_train=X_train, y_train=y_train_churn)
            
            preprocessing_obj = load_object(file_path=self.data_transformation_artifact.transformed_object_file_path)

            # --- Explicit Evaluation and Artifact Creation for Type Safety ---

            # 1. Premium Model (Regressor)
            logging.info("Evaluating Premium Model...")
            premium_model_wrapper = MyModel(preprocessing_object=preprocessing_obj, trained_model_object=premium_model)
            # ### MODIFIED ###: Call predict directly on the trained model for evaluation
            y_pred_premium = premium_model.predict(X_test)
            r2_premium = r2_score(y_test_premium, y_pred_premium)
            rmse_premium = root_mean_squared_error(y_test_premium, y_pred_premium)
            premium_metric_artifact = RegressionMetricArtifact(r2_score=r2_premium, rmse=rmse_premium)
            save_object(self.model_trainer_config.trained_model1_file_path, premium_model_wrapper)
            logging.info(f"Premium Model evaluated: R2={r2_premium:.4f}, RMSE={rmse_premium:.4f}")

            # 2. Cost Model (Regressor)
            logging.info("Evaluating Cost Model...")
            cost_metric_artifact = None
            if cost_model is not None:
                cost_model_wrapper = MyModel(preprocessing_object=preprocessing_obj, trained_model_object=cost_model)
                # ### MODIFIED ###: Call predict directly on the trained model
                y_pred_log_cost = cost_model.predict(X_test)
                y_pred_cost = np.expm1(y_pred_log_cost)
                r2_cost = r2_score(y_test_cost, y_pred_cost)
                rmse_cost = root_mean_squared_error(y_test_cost, y_pred_cost)
                cost_metric_artifact = RegressionMetricArtifact(r2_score=r2_cost, rmse=rmse_cost)
                save_object(self.model_trainer_config.trained_model2_file_path, cost_model_wrapper)
                logging.info(f"Cost Model evaluated: R2={r2_cost:.4f}, RMSE={rmse_cost:.4f}")

            # 3. Propensity Model (Classifier)
            logging.info("Evaluating Propensity Model...")
            propensity_model_wrapper = MyModel(preprocessing_object=preprocessing_obj, trained_model_object=propensity_model)
            # ### MODIFIED ###: Call predict_proba directly on the trained model
            y_pred_proba_propensity = propensity_model.predict_proba(X_test)[:, 1]
            auc_propensity = roc_auc_score(y_test_propensity, y_pred_proba_propensity)
            propensity_metric_artifact = ClassificationMetricArtifact(f1_score=None, precision_score=None, recall_score=None, auc_score=auc_propensity)
            save_object(self.model_trainer_config.trained_model3_file_path, propensity_model_wrapper)
            logging.info(f"Propensity Model evaluated: AUC={auc_propensity:.4f}")

            # 4. Churn Model (Classifier)
            logging.info("Evaluating Churn Model...")
            churn_model_wrapper = MyModel(preprocessing_object=preprocessing_obj, trained_model_object=churn_model)
            # ### MODIFIED ###: Call predict_proba directly on the trained model
            y_pred_proba_churn = churn_model.predict_proba(X_test)[:, 1]
            auc_churn = roc_auc_score(y_test_churn, y_pred_proba_churn)
            churn_metric_artifact = ClassificationMetricArtifact(f1_score=None, precision_score=None, recall_score=None, auc_score=auc_churn)
            save_object(self.model_trainer_config.trained_model4_file_path, churn_model_wrapper)
            logging.info(f"Churn Model evaluated: AUC={auc_churn:.4f}")

            # --- Create Final Artifact with Correct Types ---
            model_trainer_artifact = ModelTrainerArtifact(
                trained_model1_file_path=self.model_trainer_config.trained_model1_file_path,
                trained_model2_file_path=self.model_trainer_config.trained_model2_file_path,
                trained_model3_file_path=self.model_trainer_config.trained_model3_file_path,
                trained_model4_file_path=self.model_trainer_config.trained_model4_file_path,
                metric1_artifact=premium_metric_artifact,
                metric2_artifact=cost_metric_artifact,
                metric3_artifact=propensity_metric_artifact,
                metric4_artifact=churn_metric_artifact
            )
            return model_trainer_artifact

        except Exception as e:
            raise MyException(e, sys) from e
