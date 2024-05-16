import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from urllib.parse import urlparse
import mlflow
import mlflow.sklearn
import numpy as np
import joblib
from e2eProject.entity.config_entity import ModelEvaluationConfig
from e2eProject.utils.common import save_json
from pathlib import Path



class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig):
        self.config = config


    ##  MODEL FOR PREDICTION
    def predict_risk(input_data, model):
        # Normalize data
        scaler = StandardScaler()

        input_scaled = scaler.transform(input_data)
        risk_score = model.predict(input_scaled)
        return risk_score
    

    def eval_metrics(self,actual, pred):

        train_data = pd.read_csv(self.config.train_data_path)
        test_data = pd.read_csv(self.config.test_data_path)

        train_x = train_data.drop([self.config.target_column], axis=1)
        test_x = test_data.drop([self.config.target_column], axis=1)
        train_y = train_data[[self.config.target_column]]
        test_y = test_data[[self.config.target_column]]

        # Initialize and train the model
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(train_x, train_y)

        # Predict on the test set
        predictions = model.predict(test_x)

        # Save the model
        joblib.dump(model, 'risk_scoring_model.pkl')

        # Example new data
        new_data = pd.DataFrame({
            'Age': [50],
            'Current_Pain_Level': [5],
            'Exercise_Intensity': [2],  # Medium
            'Exercise_Duration': [30]
        })

        predicted_score = self.predict_risk(new_data,model)
        print(f'Predicted Risk Score: {predicted_score[0]}')

        # Calculate the performance
        mse = mean_squared_error(test_y, predictions)
        print(f'Mean Squared Error: {mse}')

        rmse = np.sqrt(mean_squared_error(test_y, predictions))
        mae = mean_absolute_error(test_y, predictions)
        r2 = r2_score(test_y, predictions)

        # rmse = np.sqrt(mean_squared_error(actual, pred))
        # mae = mean_absolute_error(actual, pred)
        # r2 = r2_score(actual, pred)
        return rmse, mae, r2




    def log_into_mlflow(self):

        test_data = pd.read_csv(self.config.test_data_path)
        model = joblib.load(self.config.model_path)

        test_x = test_data.drop([self.config.target_column], axis=1)
        test_y = test_data[[self.config.target_column]]

        print(self.config.mlflow_uri)
        mlflow.set_registry_uri(self.config.mlflow_uri)
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
        print(tracking_url_type_store)

        with mlflow.start_run():

            predicted_qualities = model.predict(test_x)

            (rmse, mae, r2) = self.eval_metrics(test_y, predicted_qualities)
            
            # Saving metrics as local
            scores = {"rmse": rmse, "mae": mae, "r2": r2}
            save_json(path=Path(self.config.metric_file_name), data=scores)

            mlflow.log_params(self.config.all_params)

            mlflow.log_metric("rmse", rmse)
            mlflow.log_metric("r2", r2)
            mlflow.log_metric("mae", mae)

            print(">>>>>>>>>>>>>>>>>>>>>>> pre-register model ")
            # Model registry does not work with file store
            if tracking_url_type_store != "file":

                # Register the model
                # There are other ways to use the Model Registry, which depends on the use case,
                # please refer to the doc for more information:
                # https://mlflow.org/docs/latest/model-registry.html#api-workflow
                mlflow.sklearn.log_model(model, "model", registered_model_name="ElasticnetModel")
                print(">>>>>>>>>>>>>>>>>>>>>>> registered model ")
            else:
                mlflow.sklearn.log_model(model, "model")
                print(">>>>>>>>>>>>>>>>>>>>>>> registered model ")
