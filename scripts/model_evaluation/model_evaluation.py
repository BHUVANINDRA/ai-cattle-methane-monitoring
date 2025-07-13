import lightgbm as lgb
from sklearn.metrics import mean_absolute_error, r2_score
import pandas as pd

# Load preprocessed data (from data_preprocessing.py)
df = pd.read_csv('data/processed_methane_data.csv')

# Split the data into features (X) and target (y)
X = df.drop(columns=['Methane_Production_g'])  # Features
y = df['Methane_Production_g']  # Target

# Load the trained model
model = lgb.Booster(model_file='models/methane_emission_model.txt')

# Make predictions
y_pred = model.predict(X)

# Evaluate the model
mae = mean_absolute_error(y, y_pred)
r2 = r2_score(y, y_pred)

print(f'Mean Absolute Error: {mae}')
print(f'R² Score: {r2}')

