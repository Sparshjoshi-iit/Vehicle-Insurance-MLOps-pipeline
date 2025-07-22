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
    transformed_train_file_path3:str
    transformed_test_file_path3:str
    transformed_train_file_path4:str
    transformed_test_file_path4:str

@dataclass
class ClassificationMetricArtifact:
    f1_score:float
    auc_score:float
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
    trained_model4_file_path:str
    metric1_artifact:RegressionMetricArtifact
    metric2_artifact:RegressionMetricArtifact
    metric3_artifact:ClassificationMetricArtifact
    metric4_artifact:ClassificationMetricArtifact
    
@dataclass
class ModelEvaluationArtifact:
    is_model_accepted:bool
    changed_accuracy:float
    trained_model1_path:str
    trained_model2_path:str
    trained_model3_path:str
    trained_model4_path:str
    
@dataclass
class ModelPusherArtifact:
    saved_model_dir: str
    model1_path: str
    model2_path: str
    model3_path: str
    model4_path: str