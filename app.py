from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

# Load the saved model and scaler
model = pickle.load(open('diabetes_model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

# ✅ Home route
@app.route('/')
def home():
    return render_template('index.html')

# ✅ Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        data = [
            float(request.form['Pregnancies']),
            float(request.form['Glucose']),
            float(request.form['BloodPressure']),
            float(request.form['SkinThickness']),
            float(request.form['Insulin']),
            float(request.form['BMI']),
            float(request.form['DiabetesPedigreeFunction']),
            float(request.form['Age']),
        ]

        input_data = np.array(data).reshape(1, -1)
        input_scaled = scaler.transform(input_data)
        prediction = model.predict(input_scaled)

        result = 'Diabetic' if prediction[0] == 1 else 'Not Diabetic'
        return render_template('index.html', prediction_text=f"The person is {result}")
    
# ✅ Required for Flask to run
if __name__ == '__main__':
    app.run(debug=True)
