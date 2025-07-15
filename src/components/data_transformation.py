import sys
import numpy as np
import pandas as pd
from imblearn.combine import SMOTEENN
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import PowerTransformer, OneHotEncoder
from datetime import datetime

from src.constants import TARGET_COLUMN, SCHEMA_FILE_PATH, CURRENT_YEAR,CURRENT_DATE, TARGET_COLUMN2
from src.entity.config_entity import DataTransformationConfig
from src.entity.artifact_entity import DataTransformationArtifact, DataIngestionArtifact, DataValidationArtifact
from src.exception import MyException
from src.logger import logging
from src.utils.main_utils import save_object, save_numpy_array_data, read_yaml_file

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

    def get_data_transformer_object(self) -> Pipeline:
        """
        Creates and returns a data transformer object for the data, 
        including gender mapping, dummy variable creation, column renaming,
        feature scaling, and type adjustments.
        """
        logging.info("Entered get_data_transformer_object method of DataTransformation class")

        try:
            numeric_transformer = StandardScaler()
            numeric_transformer2=StandardScaler()
            logging.info("Transformers Initialized: StandardScaler-MinMaxScaler")
            num_features = self._schema_config['numerical_columns']
            skew_transformer=PowerTransformer(method='yeo-johnson')
            logging.info("Transformers Initialized: PowerTransformer")
            
            num_features = self._schema_config['numerical_columns']
            skewed_features = self._schema_config['power_transformed_features']
            to_be_scaled_again=self._schema_config['columns_to_scale']
            processor = ColumnTransformer(
                    transformers=[
                        ("scaler", numeric_transformer, num_features),
                        ("power", skew_transformer, skewed_features),
                        ("again_scaler", numeric_transformer2,to_be_scaled_again )
                    ],
                    remainder='passthrough'
            )
            final_pipeline = Pipeline(steps=[("Preprocessor",processor)])
            logging.info("Pipeline created with StandardScaler and PowerTransformer")
            logging.info("Exited get_data_transformer_object method of DataTransformation class")
            return final_pipeline
        except Exception as e:
            raise MyException(e, sys)
    
    def create_new_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Creates new features in the DataFrame based on existing columns.
        """
        logging.info("Entered create_new_features method of DataTransformation class")
        try:
            df=df.drop(columns=['ID'])
            today = datetime(2020, 1, 1)
            df['Insurance_status'] = df['Date_lapse'].isna().astype(int)
            df['Date_lapse'] = pd.to_datetime(df['Date_lapse'],dayfirst=True, errors='coerce')
            
            df['Date_lapse'] = df['Date_lapse'].fillna(today)
            df['Date_lapse'] = df['Date_lapse'].mask(df['Date_lapse'] > today, today)
            
            logging.info("Clipping done for Date_lapse column")
            
            # Convert relevant date columns safely
            df['Date_lapse'] = pd.to_datetime(df['Date_lapse'],dayfirst=True, errors='coerce')
            df['Date_start_contract'] = pd.to_datetime(df['Date_start_contract'], dayfirst=True, errors='coerce')
            df['Date_driving_licence'] = pd.to_datetime(df['Date_driving_licence'], dayfirst=True, errors='coerce')
            df['Date_last_renewal'] = pd.to_datetime(df['Date_last_renewal'], dayfirst=True, errors='coerce')
            df['Date_next_renewal'] = pd.to_datetime(df['Date_next_renewal'], dayfirst=True, errors='coerce')

            # Age since lapse (if lapsed)
            df['Age'] = ((df['Date_lapse'] - today).dt.days // 365).fillna(0).astype(int)
            
            # Vehicle age
            df['Vehicle_age'] = CURRENT_YEAR - df['Year_matriculation'] if 'Year_matriculation' in df.columns else np.nan

            # Insurance duration (from start to lapse or today if not lapsed)
            df['Insurance_duration'] = ((df['Date_lapse'] - df['Date_start_contract']).dt.days//30).fillna(0).astype(int)
            # Licence age
            df['Licence_age'] = ((df['Date_driving_licence'] - today).dt.days // -365).fillna(0).astype(int)

            # Insurance status (1 = active, 0 = terminated)
            df['Insurance_status'] = df['Date_lapse'].isna().astype(int)

            # Time between renewals
            df['next_claim_duration'] = (
                (df['Date_next_renewal'] - df['Date_last_renewal']).dt.days
            ).fillna(0).astype(int)
            logging.info("Features: Age, Vehicle_age, Insurance_duration, Licence_age, Insurance_status, next_claim_duration created successfully")
            
            # Drop unneeded date columns
            df = df.drop(columns=['ID',
                'Date_start_contract',
                'Date_last_renewal',
                'Date_next_renewal',
                'Date_driving_licence',
                'Date_birth',
                'Date_lapse'
            ], errors='ignore')
            logging.info("Date_time columns dropped successfully")
            
            logging.info("Encoding categorical features")
            # Encode categorical features
            categorical_features = self._schema_config['categorical_columns']
            valid_categorical_cols = [col for col in categorical_features if col in df.columns]

            encoder = OneHotEncoder(sparse_output=False, drop=None, handle_unknown='ignore')
            encoded_array = encoder.fit_transform(df[valid_categorical_cols])
            encoded_df = pd.DataFrame(encoded_array, columns=encoder.get_feature_names_out(valid_categorical_cols))

            # Drop original categorical columns from df and concat encoded
            df = df.drop(columns=valid_categorical_cols)
            df_final = pd.concat([df.reset_index(drop=True), encoded_df.reset_index(drop=True)], axis=1)

            # Drop string/object columns if any remain
            df_final = df_final.drop(columns=df_final.select_dtypes(include=['object']).columns)

            return df_final
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
            
            # Load train and test data
            train_df = self.read_data(file_path=self.data_ingestion_artifact.trained_file_path)
            test_df = self.read_data(file_path=self.data_ingestion_artifact.test_file_path)
            logging.info("Train-Test data loaded")
            
            input_feature_train_df = train_df.drop(columns=[TARGET_COLUMN], axis=1)
            target_feature_train_df = train_df[TARGET_COLUMN]
            target_feature2_train_df= train_df[TARGET_COLUMN2]
            input_feature_test_df = test_df.drop(columns=[TARGET_COLUMN], axis=1)
            target_feature_test_df = test_df[TARGET_COLUMN]
            target_feature2_test_df= test_df[TARGET_COLUMN2]
            logging.info("Input and Target cols defined for both train and test df.")

            # Create new features
            input_feature_train_df = self.create_new_features(input_feature_train_df)
            input_feature_test_df = self.create_new_features(input_feature_test_df)
            logging.info("New features created successfully for train and test data")
            # Drop 'id' column if it exists
            
            # Get the data transformer object
            data_transformer = self.get_data_transformer_object()
            logging.info("Data transformer object created successfully")
            # Fit and transform the training data
            transformed_input_feature_train_df = data_transformer.fit_transform(input_feature_train_df)
            logging.info("Training data transformed successfully")
            # Transform the test data
            transformed_input_feature_test_df = data_transformer.transform(input_feature_test_df)
            logging.info("Test data transformed successfully")
            train_arr = np.c_[transformed_input_feature_train_df, np.array(target_feature_train_df)]
            test_arr = np.c_[transformed_input_feature_test_df, np.array(target_feature_test_df)]
            train_arr2=np.c_[transformed_input_feature_train_df, np.array(target_feature2_train_df)]
            test_arr2 = np.c_[transformed_input_feature_test_df, np.array(target_feature2_test_df)]
            logging.info("feature-target concatenation done for train-test df.")

            save_object(self.data_transformation_config.transformed_object_file_path, data_transformer)
            save_numpy_array_data(self.data_transformation_config.transformed_train_file_path, array=train_arr)
            save_numpy_array_data(self.data_transformation_config.transformed_test_file_path, array=test_arr)
            save_numpy_array_data(self.data_transformation_config.transformed_train_file_path2, array=train_arr2)
            save_numpy_array_data(self.data_transformation_config.transformed_test_file_path2, array=test_arr2)
            logging.info("Saving transformation object and transformed files.")
            
            logging.info("Data transformation completed successfully")
            return DataTransformationArtifact(
                transformed_object_file_path=self.data_transformation_config.transformed_object_file_path,
                transformed_train_file_path=self.data_transformation_config.transformed_train_file_path,
                transformed_test_file_path=self.data_transformation_config.transformed_test_file_path,
                transformed_train_file_path2=self.data_transformation_config.transformed_train_file_path,
                transformed_test_file_path2=self.data_transformation_config.transformed_test_file_path
            )

        except Exception as e:
            raise MyException(e, sys) from e
        
