from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler

app = Flask(__name__)

# Load the pre-trained model
try:
    model_data = joblib.load('recommender.joblib')
    clf = model_data['model']
    le_ex = model_data['le_ex']
    le_eq = model_data['le_eq']
    scaler = model_data['scaler']
    le_sex = model_data['le_sex']
    le_hypertension = model_data['le_hypertension']
    le_diabetes = model_data['le_diabetes']
    le_fitness_goal = model_data['le_fitness_goal']
    le_fitness_type = model_data['le_fitness_type']
    print("Model loaded successfully!")
except:
    print("Model not found. Please train the model first.")
    clf = None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if clf is None:
        return jsonify({'error': 'Model not loaded. Please train the model first.'})
    
    try:
        # Get form data
        sex = request.form['sex']
        age = float(request.form['age'])
        height = float(request.form['height'])
        weight = float(request.form['weight'])
        hypertension = request.form['hypertension']
        diabetes = request.form['diabetes']
        fitness_goal = request.form['fitness_goal']
        fitness_type = request.form['fitness_type']
        
        # Calculate BMI
        bmi = weight / (height ** 2)
        
        # Determine Level based on BMI
        if bmi < 18.5:
            level = 0  # Underweight
            level_name = "Underweight"
        elif bmi < 25:
            level = 1  # Normal
            level_name = "Normal"
        elif bmi < 30:
            level = 2  # Overweight
            level_name = "Overweight"
        else:
            level = 3  # Obese
            level_name = "Obese"
        
        # Encode categorical features
        sex_encoded = le_sex.transform([sex])[0]
        hypertension_encoded = le_hypertension.transform([hypertension])[0]
        diabetes_encoded = le_diabetes.transform([diabetes])[0]
        fitness_goal_encoded = le_fitness_goal.transform([fitness_goal])[0]
        fitness_type_encoded = le_fitness_type.transform([fitness_type])[0]
        
        # Create feature array
        features = np.array([[sex_encoded, age, height, weight, hypertension_encoded, 
                            diabetes_encoded, bmi, level, fitness_goal_encoded, fitness_type_encoded]])
        
        # Scale numerical features
        features[:, [1, 2, 3, 6, 7]] = scaler.transform(features[:, [1, 2, 3, 6, 7]])
        
        # Make prediction
        prediction = clf.predict(features)
        
        # Decode predictions
        exercise = le_ex.inverse_transform([prediction[0][0]])[0]
        equipment = le_eq.inverse_transform([prediction[0][1]])[0]
        
        result = {
            'success': True,
            'bmi': round(bmi, 2),
            'level': level_name,
            'exercise': exercise,
            'equipment': equipment
        }
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True, port=5000)

