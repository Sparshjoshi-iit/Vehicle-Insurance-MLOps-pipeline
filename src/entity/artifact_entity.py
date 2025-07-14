from dataclasses import dataclass


@dataclass
class DataIngestionArtifact:
    trained_file_path:str 
    test_file_path:str

@dataclass
class DataValidationArtifact:
    validation_status:bool
    message: str
    validation_report_file_path: str

@dataclass
class DataTransformationArtifact:
    transformed_object_file_path:str 
    transformed_train_file_path:str
    transformed_test_file_path:str
    transformed_train_file_path2:str
    transformed_test_file_path2:str

@dataclass
class ClassificationMetricArtifact:
    f1_score:float
    precision_score:float
    recall_score:float

@dataclass 
class RegressionMetricArtifact:
    rmse:float
    r2_score:float

@dataclass
class ModelTrainerArtifact:
    trained_model1_file_path:str
    trained_model2_file_path:str
    trained_model3_file_path:str
    metric1_artifact:ClassificationMetricArtifact
    metric2_artifact:RegressionMetricArtifact
    metric3_artifact:RegressionMetricArtifact