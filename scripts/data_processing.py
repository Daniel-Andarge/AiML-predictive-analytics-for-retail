

import pandas as pd
import os
import sys

def load_dataset(path):
    try:
        # Load the Parquet file into a DataFrame
        df = pd.read_parquet(path, engine='pyarrow')
        return df
    except FileNotFoundError as e:
        print(f"Error: {e}. The dataset file was not found.")
    except Exception as e:
        print(f"Error: {e}. An error occurred while loading the dataset.")
    return None


def save_dataset(df, output_folder, filename):
    os.makedirs(output_folder, exist_ok=True)
    output_path = os.path.join(output_folder, filename)
    df.to_parquet(output_path)
    print(f"Dataset saved to {output_path}")
    return output_path
