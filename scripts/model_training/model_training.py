import lightgbm as lgb
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split  # Import missing function
import pandas as pd

# Load preprocessed data (from data_preprocessing.py)
df = pd.read_csv('data/processed_methane_data.csv')

# Split the data into features (X) and target (y)
X = df.drop(columns=['Methane_Production_g'])  # Features
y = df['Methane_Production_g']  # Target

# Split into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Set up the LightGBM dataset
train_data = lgb.Dataset(X_train, label=y_train)
test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

# Set parameters for LightGBM
params = {
    'objective': 'regression',  # Regression problem
    'metric': 'l2',  # Mean Squared Error metric
    'boosting_type': 'gbdt',  # Gradient Boosting Decision Tree
    'num_leaves': 31,  # Number of leaves in each tree
    'learning_rate': 0.05,  # Learning rate
    'feature_fraction': 0.9,  # Fraction of features to use for each iteration
    'bagging_fraction': 0.8,  # Fraction of data to use for each iteration
    'bagging_freq': 5,  # Frequency of data bagging
    'verbose': -1  # Suppress LightGBM output
}

# Define the early stopping callback
callbacks = [lgb.early_stopping(stopping_rounds=50)]

# Train the model with early stopping via callbacks
model = lgb.train(
    params, 
    train_data, 
    valid_sets=[train_data, test_data],  # Validation sets: training and test
    valid_names=['train', 'valid'],  # Names for validation sets
    num_boost_round=1000,  # Maximum number of boosting rounds
    callbacks=callbacks  # Use early stopping
)

# Save the model
model.save_model('models/methane_emission_model.txt')

# Print the best iteration
print(f"Best Iteration: {model.best_iteration}")


import pandas as pd

df = pd.read_csv("data/processed_methane_data.csv")  # Adjust path if needed
print(df.columns.tolist())  # Print exact feature names used in training
