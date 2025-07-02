from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load the trained model
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')  # Render the HTML form

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the JSON data from the POST request
        data = request.get_json(force=True)

        # Ensure data is received correctly
        if not data:
            return jsonify({'error': 'No data provided'}), 400

        # Extract the features from the data
        features = [
            data['Year'],
            data['Present_Price'],
            data['Kms_Driven'],
            data['Owner'],
            data['Fuel_Type'],
            data['Seller_Type'],
            data['Transmission']
        ]

        # Create input dictionary with the same format used during training
        input_df = {
            'Year': [features[0]],
            'Present_Price': [features[1]],
            'Kms_Driven': [features[2]],
            'Owner': [features[3]],
            'Fuel_Type': [features[4]],
            'Seller_Type': [features[5]],
            'Transmission': [features[6]]
        }

        input_df = pd.DataFrame(input_df)
        prediction = model.predict(input_df)

        # Return the prediction as a JSON response
        return jsonify({'Predicted Selling Price': round(prediction[0], 2)})

    except Exception as e:
        # Return error message in case of exception
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
