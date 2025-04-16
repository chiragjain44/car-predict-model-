
# 🚗 Car Resale Price Prediction API

This project uses a trained machine learning model (Random Forest) to predict the resale price of a car. It includes a Flask API that takes in car details and returns the predicted price.

## 🧪 How to Run

1. **Install dependencies**
```bash
pip install flask scikit-learn pandas numpy
```

2. **Run the Flask app**
```bash
python app.py
```

3. **Make a POST request to the `/predict` endpoint**
Use a tool like Postman or `curl`, or test with the provided `request.json` file:

```bash
curl -X POST http://127.0.0.1:5000/predict -H "Content-Type: application/json" -d @request.json
```

## 📁 Files

- `model.pkl` - Trained model and preprocessor
- `app.py` - Flask API script
- `request.json` - Sample request body
- `car_data.csv` - Dataset used for training

## 🧠 Model

The model uses these features:
- Year
- Present_Price
- Kms_Driven
- Owner
- Fuel_Type
- Seller_Type
- Transmission

Prediction output: **Selling Price** of the car.

---

Made with ❤️ using Flask + scikit-learn
