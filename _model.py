import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load data
data = pd.read_csv('data.csv')

# Example columns: 'Age', 'Medical_History', 'Current_Pain_Level', 'Exercise_Intensity', 'Exercise_Duration', 'Risk_Score'

# Preprocess data: fill missing values, encode categorical data, etc.
data.fillna(data.mean(), inplace=True)  # an example of handling missing values

# Encoding categorical data - converting text to numbers
data['Exercise_Intensity'] = data['Exercise_Intensity'].map({'Low': 1, 'Medium': 2, 'High': 3})

# Split data into features and target
X = data[['Age', 'Current_Pain_Level', 'Exercise_Intensity', 'Exercise_Duration']]
y = data['Risk_Score']  # This is the target we want to predict

# Normalize data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)


##  BUILD AND TRAIN THE MODEL
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Initialize and train the model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict on the test set
predictions = model.predict(X_test)

# Calculate the performance
mse = mean_squared_error(y_test, predictions)
print(f'Mean Squared Error: {mse}')


##  MODEL FOR PREDICTION
def predict_risk(input_data):
    input_scaled = scaler.transform(input_data)
    risk_score = model.predict(input_scaled)
    return risk_score

# Example new data
new_data = pd.DataFrame({
    'Age': [50],
    'Current_Pain_Level': [5],
    'Exercise_Intensity': [2],  # Medium
    'Exercise_Duration': [30]
})

predicted_score = predict_risk(new_data)
print(f'Predicted Risk Score: {predicted_score[0]}')


## DEPLOYMENT
import joblib

# Save the model
joblib.dump(model, 'risk_scoring_model.pkl')

# Load the model in another application or server script
loaded_model = joblib.load('risk_scoring_model.pkl')




##GENERATE DATA SET
import pandas as pd
import random

# Random seed for reproducibility
random.seed(42)

# Generate sample data
data = {
    'Patient_ID': range(1, 101),
    'Age': [random.randint(18, 80) for _ in range(100)],
    'Gender': [random.choice(['Male', 'Female']) for _ in range(100)],
    'Medical_History': [random.choice(['None', 'Cardiovascular', 'Orthopedic', 'Diabetes', 'Chronic Pain']) for _ in range(100)],
    'Current_Pain_Level': [random.randint(1, 10) for _ in range(100)],
    'Exercise_Type': [random.choice(['Stretching', 'Strength', 'Cardio', 'Flexibility']) for _ in range(100)],
    'Exercise_Intensity': [random.choice(['Low', 'Medium', 'High']) for _ in range(100)],
    'Exercise_Duration': [random.randint(10, 60) for _ in range(100)]
}

df = pd.DataFrame(data)

# Save to CSV
csv_file_path = '/mnt/data/Physical_Therapy_Patient_Data.csv'
df.to_csv(csv_file_path, index=False)
