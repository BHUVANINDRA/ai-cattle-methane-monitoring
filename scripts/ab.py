import pandas as pd

# Load dataset
data = pd.read_csv('data/preprocessed_data.csv')

# Parameters to check
parameters = ['Age_Years', 'Weight_kg', 'Fiber_Content_%', 'Carbohydrates_Content_%']

# Get the range of each parameter
for param in parameters:
    print(f"{param}: Min = {data[param].min()}, Max = {data[param].max()}")
    
