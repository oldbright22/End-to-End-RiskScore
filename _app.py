from flask import Flask, render_template, request, jsonify
import os 
import numpy as np
import pandas as pd

from flask_sqlalchemy import SQLAlchemy

##from e2eProject.pipeline.prediction import PredictionPipeline

app = Flask(__name__) # initializing a flask app


##@app.route('/',methods=['GET'])               # route to display the home page
##def homePage():
##    return render_template("index.html")


##@app.route('/train',methods=['GET'])          # route to train the pipeline
##def training():
##    os.system("python main.py")
##    return "Training Successful!" 


app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///site.db'
db = SQLAlchemy(app)

class Patient(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    age = db.Column(db.Integer, nullable=False)
    medical_conditions = db.Column(db.String(200), default="")

# Initialize the database
db.create_all()


@app.route('/')
def index():
    return "Hello, World!"


## def calculate_risk(age, medical_conditions, current_pain_level, exercise_intensity, exercise_duration):
##     risk_score = age / 10 + current_pain_level * 1.5
##     if 'high-risk' in medical_conditions.lower():
##         risk_score += 20
##     if exercise_intensity == 'High':
##         risk_score += 10
##     return risk_score


##def calculate_improvement(pre_pain, post_pain):
##    return pre_pain - post_pain


def calculate_risk(age, medical_conditions, current_pain_level, exercise_intensity, exercise_duration):
    base_risk = 0

    # Age factor: Older patients generally have a higher risk of complications.
    if age > 50:
        base_risk += 5
    elif age > 65:
        base_risk += 10

    # Medical condition factor: Certain conditions increase the risk during physical activities.
    if 'cardiovascular' in medical_conditions.lower():
        base_risk += 15
    if 'orthopedic' in medical_conditions.lower():
        base_risk += 10
    if 'high-risk' in medical_conditions.lower():
        base_risk += 20

    # Current pain level: Higher initial pain levels might indicate more severe conditions.
    if current_pain_level > 7:
        base_risk += 10

    # Exercise intensity and duration factors.
    intensity_factor = {'Low': 0, 'Medium': 5, 'High': 10}
    base_risk += intensity_factor[exercise_intensity]

    # Longer durations add more risk due to fatigue and strain.
    if exercise_duration > 60:
        base_risk += 10

    return base_risk


def calculate_improvement(pre_pain, post_pain, intensity):
    improvement = (pre_pain - post_pain) * 1.1  # Base improvement factor
    intensity_adjustment = {'Low': 1.0, 'Medium': 1.05, 'High': 1.1}
    improvement *= intensity_adjustment[intensity]
    return max(0, improvement)  # Ensure non-negative improvement scores


@app.route('/score/risk', methods=['POST'])
def get_risk_score():
    data = request.get_json()
    score = calculate_risk(data['age'], data['medical_conditions'], data['current_pain_level'], data['exercise_intensity'], data['exercise_duration'])
    return jsonify({'risk_score': score})


@app.route('/score/improvement', methods=['POST'])
def get_improvement_score():
    data = request.get_json()
    score = calculate_improvement(data['pre_pain'], data['post_pain'], data['intensity'])
    return jsonify({'improvement_score': score})


@app.route('/patients', methods=['POST'])
def add_patient():
    data = request.get_json()
    patient = Patient(name=data['name'], age=data['age'], medical_conditions=data['medical_conditions'])
    db.session.add(patient)
    db.session.commit()
    return jsonify({'message': 'Patient added'}), 201


@app.route('/patients/<int:patient_id>', methods=['GET'])
def get_patient(patient_id):
    patient = Patient.query.get_or_404(patient_id)
    return jsonify({'name': patient.name, 'age': patient.age, 'medical_conditions': patient.medical_conditions})


@app.route('/form')
def form():
    return app.send_from_directory('.', 'index.html')


##if __name__ == '__main__':
##    app.run(debug=True)


if __name__ == "__main__":
  # app.run(host="0.0.0.0", port = 8080, debug=True)
	app.run(host="0.0.0.0", port = 8080)
     