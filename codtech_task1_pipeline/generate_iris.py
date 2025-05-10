from sklearn.datasets import load_iris
import pandas as pd

# Load iris dataset
data = load_iris(as_frame=True)
df = data.frame

# Save to CSV
df.to_csv('iris.csv', index=False)
print("iris.csv has been created.")
