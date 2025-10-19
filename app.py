import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Load the dataset
data = pd.read_csv('dataset.csv')

# drop Diet and Recommendation columns
data = data.drop(columns=['Diet', 'Recommendation'])

# encode specified columns 'Sex', 'Hypertension', 'Diabetes', 'Fitness Goal', 'Fitness Type'
for column in ['Sex', 'Hypertension', 'Diabetes', 'Fitness Goal', 'Fitness Type']:
	data[column] = LabelEncoder().fit_transform(data[column])

# encode 'Level' column with custom mapping
level_mapping = {'Underweight': 0, 'Normal': 1, 'Overweight': 2, 'Obuse': 3}
data['Level'] = data['Level'].map(level_mapping)

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

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
	X, y, test_size=0.2, random_state=42
)

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

# Evaluate model with accuracy, precision, recall, and F1-score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
y_pred = clf.predict(X_test)

