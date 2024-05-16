import os
from e2eProject import logger
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
from e2eProject.entity.config_entity import DataTransformationConfig



class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config

   
    def train_test_spliting(self):
        data = pd.read_csv(self.config.data_path)

        # Example columns: 'Age', 'Medical_History', 'Current_Pain_Level', 'Exercise_Intensity', 'Exercise_Duration', 'Risk_Score'

        # Preprocess data: fill missing values, encode categorical data, etc.
        data.fillna(data.mean(), inplace=True)  # handling missing values

        # Encoding categorical data - converting text to numbers
        data['Exercise_Intensity'] = data['Exercise_Intensity'].map({'Low': 1, 'Medium': 2, 'High': 3})

        # Split data into features and target
        X = data[['Age', 'Current_Pain_Level', 'Exercise_Intensity', 'Exercise_Duration']]
        y = data['Risk_Score']  # This is the target we want to predict

        # Normalize data
        scaler = StandardScaler()  #Standarize features by removing the mean
        X_scaled = scaler.fit_transform(X)

        # Split the data into training and test sets. (0.75, 0.25) split.
        # train, test = train_test_split(data)

        # Split the data into training and test sets
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

        X_train.to_csv(os.path.join(self.config.root_dir, "train_x.csv"),index = False)
        X_test.to_csv(os.path.join(self.config.root_dir, "test_x.csv"),index = False)
        y_train.to_csv(os.path.join(self.config.root_dir, "train_y.csv"),index = False)
        y_test.to_csv(os.path.join(self.config.root_dir, "test_y.csv"),index = False)

        logger.info("Splited data into training and test sets")
        logger.info(X_train.shape)
        logger.info(X_test.shape)
        logger.info(y_train.shape)
        logger.info(y_test.shape)

        print(X_train.shape)
        print(X_test.shape)
        print(y_train.shape)
        print(y_test.shape)
        

    #You can perform all kinds of EDA in ML cycle here before passing this data to the model