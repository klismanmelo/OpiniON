import os
import pandas as pd

def load_file():
    csv_file = os.path.join("./", "sentimentos.csv")
    data = pd.read_csv(csv_file)
    print(data.head())
    return data