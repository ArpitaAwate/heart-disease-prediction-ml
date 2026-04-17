import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# Load dataset
data = pd.read_csv('data/heart.csv')

# Ensure the features match the model input requirements
X = data.drop('target', axis=1)  # Features (13 features)
y = data['target']              # Target (0 or 1)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
print('Model Accuracy:', accuracy_score(y_test, y_pred))

# Save the trained model
joblib.dump(model, 'model/heart_model.pkl')
