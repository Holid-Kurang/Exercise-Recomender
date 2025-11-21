import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

print("Loading dataset...")
# Load the dataset
data = pd.read_csv('dataset.csv')

# drop Diet and Recommendation columns
data = data.drop(columns=['Diet', 'Recommendation'])

print("Encoding categorical features...")
# Create label encoders for each categorical column
le_sex = LabelEncoder()
le_hypertension = LabelEncoder()
le_diabetes = LabelEncoder()
le_fitness_goal = LabelEncoder()
le_fitness_type = LabelEncoder()

# Fit and transform each column
data['Sex'] = le_sex.fit_transform(data['Sex'])
data['Hypertension'] = le_hypertension.fit_transform(data['Hypertension'])
data['Diabetes'] = le_diabetes.fit_transform(data['Diabetes'])
data['Fitness Goal'] = le_fitness_goal.fit_transform(data['Fitness Goal'])
data['Fitness Type'] = le_fitness_type.fit_transform(data['Fitness Type'])

# encode 'Level' column with custom mapping
level_mapping = {'Underweight': 0, 'Normal': 1, 'Overweight': 2, 'Obuse': 3}
data['Level'] = data['Level'].map(level_mapping)

print("Normalizing numerical features...")
# normalize Age, Height, Weight, BMI, Level columns
scaler = StandardScaler()
data[['Age', 'Height', 'Weight', 'BMI', 'Level']] = scaler.fit_transform(data[['Age', 'Height', 'Weight', 'BMI', 'Level']])

# Fitur dan label
feature_cols = ['Sex','Age','Height','Weight','Hypertension','Diabetes','BMI','Level','Fitness Goal','Fitness Type']
X = data[feature_cols].values

# Encode target labels
le_ex = LabelEncoder().fit(data['Exercises'])
le_eq = LabelEncoder().fit(data['Equipment'])
y_ex = le_ex.transform(data['Exercises'])
y_eq = le_eq.transform(data['Equipment'])
y = np.column_stack([y_ex, y_eq])

print("Splitting data into train and test sets...")
# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
	X, y, test_size=0.2, random_state=42
)

print("Training model...")
# Model dasar
base = RandomForestClassifier(
	n_estimators=300,
	class_weight='balanced_subsample',
	n_jobs=-1,
	random_state=42
)
clf = MultiOutputClassifier(base)

# Train
clf.fit(X_train, y_train)

print("Evaluating model...")
# Evaluate model with accuracy, precision, recall, and F1-score
y_pred = clf.predict(X_test)

# Calculate metrics for each output
for i, target in enumerate(['Exercises', 'Equipment']):
    accuracy = accuracy_score(y_test[:, i], y_pred[:, i])
    precision = precision_score(y_test[:, i], y_pred[:, i], average='weighted', zero_division=0)
    recall = recall_score(y_test[:, i], y_pred[:, i], average='weighted', zero_division=0)
    f1 = f1_score(y_test[:, i], y_pred[:, i], average='weighted', zero_division=0)
    
    print(f"\n{target} Prediction:")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1-score: {f1:.4f}")

print("\nSaving model...")
# Save the model and all necessary encoders
model_data = {
    'model': clf,
    'le_ex': le_ex,
    'le_eq': le_eq,
    'scaler': scaler,
    'le_sex': le_sex,
    'le_hypertension': le_hypertension,
    'le_diabetes': le_diabetes,
    'le_fitness_goal': le_fitness_goal,
    'le_fitness_type': le_fitness_type
}

joblib.dump(model_data, 'recommender.joblib')
print("Model saved successfully as 'recommender.joblib'!")
