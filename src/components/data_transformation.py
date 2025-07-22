import sys
import numpy as np
import pandas as pd
from imblearn.combine import SMOTEENN
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import PowerTransformer, OneHotEncoder
from datetime import datetime
from sklearn.impute import SimpleImputer
from src.constants import TARGET_COLUMN, SCHEMA_FILE_PATH, CURRENT_YEAR,CURRENT_DATE, TARGET_COLUMN2,TARGET_COLUMN3,TARGET_COLUMN4
from src.entity.config_entity import DataTransformationConfig
from src.entity.artifact_entity import DataTransformationArtifact, DataIngestionArtifact, DataValidationArtifact
from src.exception import MyException
from src.logger import logging
from src.utils.main_utils import save_object, save_numpy_array_data, read_yaml_file
from scipy.sparse import hstack

class DataTransformation:
    def __init__(self, data_ingestion_artifact: DataIngestionArtifact,
                 data_transformation_config: DataTransformationConfig,
                 data_validation_artifact: DataValidationArtifact):
        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_transformation_config = data_transformation_config
            self.data_validation_artifact = data_validation_artifact
            self._schema_config = read_yaml_file(file_path=SCHEMA_FILE_PATH)
        except Exception as e:
            raise MyException(e, sys)

    @staticmethod
    def read_data(file_path) -> pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise MyException(e, sys)

    def get_data_transformer_object(self,X):
        """
        Creates and returns a data transformer object for the data, 
        including gender mapping, dummy variable creation, column renaming,
        feature scaling, and type adjustments.
        """
        logging.info("Entered get_data_transformer_object method of DataTransformation class")

        try:

            categorical_features =X.select_dtypes(include=['object', 'category']).columns
            for col in categorical_features:
                X[col] = X[col].astype(str)
            numerical_features = X.select_dtypes(include=np.number).columns    
            
            num_features = self._schema_config['numerical_columns']
            skew_transformer=PowerTransformer(method='yeo-johnson')
            
            numerical_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median')), # Use median for robustness to outliers
                ('scaler', StandardScaler())
            ])

            categorical_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('onehot', OneHotEncoder(handle_unknown='ignore'))
            ])
            
            # Create a preprocessor object using ColumnTransformer
            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', numerical_transformer, numerical_features),
                    ('cat', categorical_transformer, categorical_features)
                ],
                remainder='passthrough' # Keep other columns (if any)
            )
            final_pipeline1=Pipeline(steps=[("PreProcessor",preprocessor)])
            logging.info("Pipeline created with StandardScaler and PowerTransformer")
            logging.info("Exited get_data_transformer_object method of DataTransformation class")
            return final_pipeline1
        except Exception as e:
            raise MyException(e, sys)
    
    def create_new_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Creates new features in the DataFrame based on existing columns.
        """
        logging.info("Entered create_new_features method of DataTransformation class")
        try:
            
            df = df.copy()

            # --- 1. Date and Time Feature Engineering ---
            date_cols = [
                'Date_start_contract', 'Date_last_renewal', 'Date_next_renewal',
                'Date_birth', 'Date_driving_licence', 'Date_lapse'
            ]
            for col in date_cols:
                df[col] = pd.to_datetime(df[col], dayfirst=True, errors='coerce')

            ref_date = df['Date_last_renewal'].max() or datetime.now()
            
            df['Age'] = ((ref_date - df['Date_birth']).dt.days / 365.25).astype(int)
            df['Licence_age'] = ((ref_date - df['Date_driving_licence']).dt.days / 365.25).astype(int)
            df['Vehicle_age'] = ref_date.year - df['Year_matriculation']
            df['Contract_duration_days'] = (df['Date_last_renewal'] - df['Date_start_contract']).dt.days
            df['Claim_Propensity'] = (df['Cost_claims_year'] > 0).astype(int)
            df['Customer_Churn'] = (~df['Date_lapse'].isna()).astype(int)

            df = df.sort_values(by=['ID', 'Date_last_renewal'])

            # Lag Features (Previous Year's Data)
            df['Previous_Premium'] = df.groupby('ID')['Premium'].shift(1)
            df['Previous_N_claims_year'] = df.groupby('ID')['N_claims_year'].shift(1)
            df['Previous_Cost_claims_year'] = df.groupby('ID')['Cost_claims_year'].shift(1)

            # Cumulative Features (History *before* the current year)
            df['Cumulative_Claims_Lagged'] = df.groupby('ID')['N_claims_year'].cumsum().shift(1)
            df['Cumulative_Premium_Lagged'] = df.groupby('ID')['Premium'].cumsum().shift(1)
            df['Cumulative_Cost_Claims_Lagged'] = df.groupby('ID')['Cost_claims_year'].cumsum().shift(1)
            
            # Rolling Aggregates (Trends over last 2 years, *excluding* the current year)
            df['Premium_2Y_Avg_Lagged'] = df.groupby('ID')['Premium'].rolling(window=2, min_periods=1).mean().shift(1).reset_index(0, drop=True)
            df['Claims_2Y_Sum_Lagged'] = df.groupby('ID')['N_claims_year'].rolling(window=2, min_periods=1).sum().shift(1).reset_index(0, drop=True)

            # Trend Features (Change from year N-2 to N-1)
            df['Premium_Change_Lagged'] = df.groupby('ID')['Premium'].diff(1).shift(1)

            df['Claim_Cost_per_Premium_History'] = df['Cumulative_Cost_Claims_Lagged'] / (df['Cumulative_Premium_Lagged'] + 1e-6)
            num_previous_years = df.groupby('ID').cumcount()
            df['Claim_Frequency_History'] = df['Cumulative_Claims_Lagged'] / num_previous_years.replace(0, 1)

            df['Age_x_Licence_age'] = df['Age'] * df['Licence_age']
            df['Seniority_x_Policies'] = df['Seniority'] * df['Policies_in_force']
            df['Value_per_Weight'] = df['Value_vehicle'] / (df['Weight'] + 1e-6)
            
            # --- 5. Final Cleanup ---
            cols_to_drop = date_cols + ['Year_matriculation', 'R_Claims_history']
            df = df.drop(columns=cols_to_drop, errors='ignore')
            
            # Fill NaNs created by shift/rolling operations.
            lagged_cols = [col for col in df.columns if '_Lagged' in col or 'Previous_' in col or '_History' in col]
            for col in lagged_cols:
                df[col] = df[col].fillna(0)
                    
            return df
            
        except Exception as e:
            raise MyException(e, sys)   
    
    def initiate_data_transformation(self) -> DataTransformationArtifact:
        """
        Initiates the data transformation component for the pipeline.
        """
        try:
            logging.info("Data Transformation Started !!!")
            if not self.data_validation_artifact.validation_status:
                raise Exception(self.data_validation_artifact.message)
            
            # Load original train and test data
            train_df = self.read_data(file_path=self.data_ingestion_artifact.trained_file_path)
            test_df = self.read_data(file_path=self.data_ingestion_artifact.test_file_path)
            logging.info("Train-Test data loaded")
            
            # STEP 1: Apply feature engineering to create new columns
            logging.info("Applying feature engineering to create new columns...")
            train_df_featured = self.create_new_features(train_df)
            test_df_featured = self.create_new_features(test_df)
            logging.info("Feature engineering complete.")

            # STEP 2: Define targets and separate features from the engineered dataframes
            all_target_columns = [TARGET_COLUMN, TARGET_COLUMN2, 'Claim_Propensity', 'Customer_Churn']
            input_feature_train_df = train_df_featured.drop(columns=all_target_columns, axis=1)
            input_feature_test_df = test_df_featured.drop(columns=all_target_columns, axis=1)
            logging.info("Input features have been separated from all target columns.")

            # STEP 3: Preprocess the feature DataFrames
            data_preprocessor = self.get_data_transformer_object(input_feature_train_df.copy())
            transformed_input_train_features = data_preprocessor.fit_transform(input_feature_train_df)
            transformed_input_test_features = data_preprocessor.transform(input_feature_test_df)
            logging.info("Sparse feature matrices created successfully.")

            # STEP 4: Combine transformed features with targets and save all artifacts
            
            # For Premium model
            target_premium_train = train_df_featured[TARGET_COLUMN].values.reshape(-1, 1)
            target_premium_test = test_df_featured[TARGET_COLUMN].values.reshape(-1, 1)
            train_arr_premium = hstack([transformed_input_train_features, target_premium_train])
            test_arr_premium = hstack([transformed_input_test_features, target_premium_test])
            save_object(self.data_transformation_config.transformed_train_file_path, train_arr_premium)
            save_object(self.data_transformation_config.transformed_test_file_path, test_arr_premium)
            logging.info("Saved Premium model train and test arrays.")

            # For Claim Cost model
            target_cost_train = train_df_featured[TARGET_COLUMN2].values.reshape(-1, 1)
            target_cost_test = test_df_featured[TARGET_COLUMN2].values.reshape(-1, 1)
            train_arr_cost = hstack([transformed_input_train_features, target_cost_train])
            test_arr_cost = hstack([transformed_input_test_features, target_cost_test])
            save_object(self.data_transformation_config.transformed_train_file_path2, train_arr_cost)
            save_object(self.data_transformation_config.transformed_test_file_path2, test_arr_cost)
            logging.info("Saved Claim Cost model train and test arrays.")

            # For Claim Propensity model
            target_propensity_train = train_df_featured['Claim_Propensity'].values.reshape(-1, 1)
            target_propensity_test = test_df_featured['Claim_Propensity'].values.reshape(-1, 1)
            train_arr_propensity = hstack([transformed_input_train_features, target_propensity_train])
            test_arr_propensity = hstack([transformed_input_test_features, target_propensity_test])
            save_object(self.data_transformation_config.transformed_train_file_path3, train_arr_propensity)
            save_object(self.data_transformation_config.transformed_test_file_path3, test_arr_propensity)
            logging.info("Saved Claim Propensity model train and test arrays.")

            # For Customer Churn model
            target_churn_train = train_df_featured['Customer_Churn'].values.reshape(-1, 1)
            target_churn_test = test_df_featured['Customer_Churn'].values.reshape(-1, 1)
            train_arr_churn = hstack([transformed_input_train_features, target_churn_train])
            test_arr_churn = hstack([transformed_input_test_features, target_churn_test])
            save_object(self.data_transformation_config.transformed_train_file_path4, train_arr_churn)
            save_object(self.data_transformation_config.transformed_test_file_path4, test_arr_churn)
            logging.info("Saved Customer Churn model train and test arrays.")

            # Save the preprocessor object
            save_object(self.data_transformation_config.transformed_object_file_path, data_preprocessor)
            logging.info("Data transformation object saved.")
            
            logging.info("Data transformation completed successfully")
            
            # Create the final artifact with all file paths
            data_transformation_artifact = DataTransformationArtifact(
                transformed_object_file_path=self.data_transformation_config.transformed_object_file_path,
                transformed_train_file_path=self.data_transformation_config.transformed_train_file_path,
                transformed_test_file_path=self.data_transformation_config.transformed_test_file_path,
                transformed_train_file_path2=self.data_transformation_config.transformed_train_file_path2,
                transformed_test_file_path2=self.data_transformation_config.transformed_test_file_path2,
                transformed_train_file_path3=self.data_transformation_config.transformed_train_file_path3,
                transformed_test_file_path3=self.data_transformation_config.transformed_test_file_path3,
                transformed_train_file_path4=self.data_transformation_config.transformed_train_file_path4,
                transformed_test_file_path4=self.data_transformation_config.transformed_test_file_path4,
            )
            return data_transformation_artifact

        except Exception as e:
            raise MyException(e, sys) from e
        
        
        