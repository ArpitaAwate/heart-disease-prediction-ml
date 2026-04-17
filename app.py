from flask import Flask, request, render_template
import joblib

# Initialize Flask app
app = Flask(__name__)

# Load the trained model
model = joblib.load('model/heart_model.pkl')

# Home route to render the HTML form
@app.route('/', methods=['GET', 'POST'])
def home():
    result = None
    if request.method == 'POST':
        # Collect form data
        feature_order = ['age', 'restbp', 'chol', 'fbs', 'restecg', 'thalach', 'exang',
                         'oldpeak', 'slope', 'ca', 'thal', 'sex', 'cp']
        try:
            data = [float(request.form[feature]) for feature in feature_order]

            # Make prediction
            prediction = model.predict([data])
            result = 'Heart Disease' if prediction[0] == 1 else 'No Heart Disease'
        except Exception as e:
            result = f"Error: {e}"

    return render_template('index.html', result=result)

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)