import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# Load the dataset 
df = pd.read_csv('data/updated_synthetic_methane_dataset.csv')

# Encode categorical variables using LabelEncoder
categorical_columns = ['Cattle_Breed', 'Activity_Type', 'State', 'Supplement', 'Medical_Intervention']
label_encoders = {}

for col in categorical_columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Split the data into features (X) and target (y)
X = df.drop(columns=['Methane_Production_g'])  # Features
y = df['Methane_Production_g']  # Target

# Split the data into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Optionally, save the preprocessed data (optional, can skip this part if not needed)
df.to_csv('data/processed_methane_data.csv', index=False)

# Print shapes to confirm the split
print(f'Training data shape: {X_train.shape}')
print(f'Testing data shape: {X_test.shape}')
