import pandas as pd
import os
import joblib

from e2eProject import logger
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor

from sklearn.linear_model import ElasticNet
import joblib
from e2eProject.entity.config_entity import ModelTrainerConfig


class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config
        
   
    def train(self):
        train_x_data = pd.read_csv(self.config.train_x_data_path)
        train_y_data = pd.read_csv(self.config.train_y_data_path)
        test_x_data = pd.read_csv(self.config.test_x_data_path)
        test_y_data = pd.read_csv(self.config.test_y_data_path)

        train_x = train_x_data.drop([self.config.target_column], axis=1)
        test_x = test_x_data.drop([self.config.target_column], axis=1)
        train_y = train_y_data[[self.config.target_column]]
        test_y = test_y_data[[self.config.target_column]]

        lr = ElasticNet(alpha=self.config.alpha, l1_ratio=self.config.l1_ratio, random_state=42)
        lr.fit(train_x, train_y)

        joblib.dump(lr, os.path.join(self.config.root_dir, self.config.model_name))
