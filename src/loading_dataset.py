import pandas as pd

# load a parquet file into a pandas DataFrame
def load_parquet_file(file_path):
    return pd.read_parquet(file_path)

df = load_parquet_file("dataset/train-00000-of-00001.parquet")
print(df.columns)