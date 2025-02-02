import pandas as pd
import xgboost as xgb
import joblib
from sklearn.model_selection import train_test_split
from datetime import datetime

# Load dataset
url = "https://raw.githubusercontent.com/ageron/handson-ml2/master/datasets/housing/housing.csv"
data = pd.read_csv(url)

# Select relevant features
data = data[['median_income', 'total_rooms', 'housing_median_age', 'median_house_value']]
data.dropna(inplace=True)

# Split data into training and testing sets
X = data.drop(columns=['median_house_value'])
y = data['median_house_value']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train XGBoost model
model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, max_depth=5)
model.fit(X_train, y_train)

# Generate timestamp for model versioning
timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
model_filename = f"house_price_model_{timestamp}.pkl"

# Save model
joblib.dump(model, model_filename)

print(f"Model training complete! Saved as '{model_filename}'")
