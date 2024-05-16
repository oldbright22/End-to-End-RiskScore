from e2eProject.constants import *
from e2eProject.utils.common import read_yaml, create_directories
from e2eProject.entity.config_entity import (DataIngestionConfig,
                                             DataValidationConfig,
                                             DataTransformationConfig,
                                             ModelTrainerConfig,
                                             ModelEvaluationConfig)

class ConfigurationManager:
    def __init__(
        self,
        config_filepath = CONFIG_FILE_PATH,
        params_filepath = PARAMS_FILE_PATH,
        schema_filepath = SCHEMA_FILE_PATH):

        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)
        self.schema = read_yaml(schema_filepath)

        create_directories([self.config.artifacts_root])

    
    def get_data_validation_config(self) -> DataValidationConfig:
        config = self.config.data_validation
        schema = self.schema.COLUMNS

        create_directories([config.root_dir])

        data_validation_config = DataValidationConfig(
            root_dir=config.root_dir,
            STATUS_FILE=config.STATUS_FILE,
            unzip_data_dir = config.unzip_data_dir,
            all_schema=schema,
        )

        return data_validation_config
    

    def get_data_transformation_config(self) -> DataTransformationConfig:
        config = self.config.data_transformation

        create_directories([config.root_dir])

        data_transformation_config = DataTransformationConfig(
            root_dir=config.root_dir,
            data_path=config.data_path,
        )

        return data_transformation_config
    

    def get_model_trainer_config(self) -> ModelTrainerConfig:
        config = self.config.model_trainer
        params = self.params.ElasticNet
        schema =  self.schema.TARGET_COLUMN

        create_directories([config.root_dir])

        model_trainer_config = ModelTrainerConfig(
            root_dir=config.root_dir,
            train_x_data_path = config.train_x_data_path,
            train_y_data_path = config.train_y_data_path,
            test_x_data_path = config.test_x_data_path,
            test_y_data_path = config.test_y_data_path,
            model_name = config.model_name,
            alpha = params.alpha,
            l1_ratio = params.l1_ratio,
            target_column = schema.name    
        )

        return model_trainer_config
    


    def get_model_evaluation_config(self) -> ModelEvaluationConfig:
        config = self.config.model_evaluation
        params = self.params.ElasticNet
        schema =  self.schema.TARGET_COLUMN

        create_directories([config.root_dir])

        model_evaluation_config = ModelEvaluationConfig(
            root_dir=config.root_dir,
            test_data_path=config.test_data_path,
            model_path = config.model_path,
            all_params=params,
            metric_file_name = config.metric_file_name,
            target_column = schema.name,
            mlflow_uri = config.mlflow_uri
        )

        return model_evaluation_config