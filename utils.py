import glob
import os

import pandas as pd


def merge_csv_files(folder_path, output_file="merged_output.csv"):
    """
    Merges all CSV files in the specified folder into a single CSV file.

    Parameters:
    - folder_path: str, the path to the folder containing CSV files.
    - output_file: str, the name of the output CSV file (default is 'merged_output.csv').

    Returns:
    - merged_df: pandas.DataFrame, the concatenated DataFrame of all CSV files.
    """
    # Use glob to find all CSV files in the folder
    csv_files = glob.glob(os.path.join(folder_path, "*.csv"))

    # Read each CSV file into a DataFrame and store in a list
    dfs = [pd.read_csv(file) for file in csv_files]

    # Concatenate all DataFrames into one
    merged_df = pd.concat(dfs, ignore_index=True)

    # Save the merged DataFrame to a new CSV file
    merged_df.to_csv(os.path.join(folder_path, output_file), index=False)

    print(f"All CSV files have been merged successfully into {output_file}!")
    return merged_df


def split_csv_file(input_file, train_file, test_file, train_frac=0.85, random_state=42):
    """
    Splits the CSV file into two files: one with train_frac percentage of the rows,
    and another with the rest.

    Parameters:
        input_file (str): Path to the input CSV file.
        train_file (str): Path to save the 80% (or train_frac) split CSV.
        test_file (str): Path to save the remaining 20% CSV.
        train_frac (float): Fraction of data to be included in the train file (default is 0.8).
        random_state (int): Seed for reproducibility.
    """
    # Read the CSV file
    df = pd.read_csv(input_file)

    # Randomly sample train_frac portion for the training set
    train_df = df.sample(frac=train_frac, random_state=random_state)

    # The rest of the data will be in the test set
    test_df = df.drop(train_df.index)

    # Write the resulting DataFrames to CSV files
    train_df.to_csv(train_file, index=False)
    test_df.to_csv(test_file, index=False)

    print(f"Data split complete: {len(train_df)} rows in '{train_file}' and {len(test_df)} rows in '{test_file}'.")
