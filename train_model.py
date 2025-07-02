
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
import pickle

# Load the dataset
df = pd.read_csv('car_data.csv')

# Features and target variable
X = df.drop(columns=['Selling_Price', 'Car_Name'])
y = df['Selling_Price']

# Identify categorical and numerical columns
categorical_cols = ['Fuel_Type', 'Seller_Type', 'Transmission']
numerical_cols = ['Year', 'Present_Price', 'Kms_Driven', 'Owner']

# Preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(drop='first'), categorical_cols),
        ('num', StandardScaler(), numerical_cols)
    ]
)

# Full pipeline with the RandomForest model
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit the model
pipeline.fit(X_train, y_train)
y_pred=pipeline.predict(X_test)
from sklearn.metrics import mean_absolute_error,mean_squared_error
mean_absolute_error(y_test,y_pred)
mean_squared_error(y_test,y_pred)


# Save the model to a file
with open('model.pkl', 'wb') as model_file:
    pickle.dump(pipeline, model_file)

print("Model saved as 'model.pkl'")
