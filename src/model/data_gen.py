import pandas as pd
from sklearn.datasets import load_wine
import os

def generate_data():
    os.makedirs('data', exist_ok=True)
    wine = load_wine()
    df = pd.DataFrame(data=wine.data, columns=wine.feature_names)
    df['target'] = wine.target
    
    # Save to CSV
    df.to_csv('data/wine_quality.csv', index=False)
    print("Data generated at data/wine_quality.csv")

if __name__ == "__main__":
    generate_data()
