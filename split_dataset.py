import os

import numpy as np
import pandas as pd


def split_train_test_csv_file(data_frame, train_file, test_file, train_frac=0.85, random_state=42):
    """
    Splits the CSV file into two files: one with train_frac percentage of the rows,
    and another with the rest.

    Parameters:
        train_file (str): Path to save the 80% (or train_frac) split CSV.
        test_file (str): Path to save the remaining 20% CSV.
        train_frac (float): Fraction of data to be included in the train file (default is 0.8).
        random_state (int): Seed for reproducibility.
    """

    # Randomly sample train_frac portion for the training set
    train_df = data_frame.sample(frac=train_frac, random_state=random_state)

    # The rest of the data will be in the test set
    test_df = data_frame.drop(train_df.index)

    # Write the resulting DataFrames to CSV files
    train_df.to_csv(train_file, index=False)
    test_df.to_csv(test_file, index=False)

    print(f"Data split complete: {len(train_df)} rows in '{train_file}' and {len(test_df)} rows in '{test_file}'.")
    return train_df, test_df


def split_into_k_files(data_frame, folder_path, sensitive_column_1='sex',
                       sensitive_column_2='health', sensitive_column_3=None,
                       output_column='Probability', save_file=True):
    """
    Split the dataset based on sensitive columns and return a list of dictionaries with file info
    
    :param data_frame: data loaded from csv file
    :param folder_path: folder to save csv files
    :param sensitive_column_1: first sensitive column to split on
    :param sensitive_column_2: second sensitive column to split on
    :param sensitive_column_3: optional third sensitive column to split on
    :param output_column: column with output probabilities
    :param save_file: whether to save the files to disk
    :return: list of dictionaries, each containing file info and length
    """

    result_files = []
    for sensitive_1_i in data_frame[sensitive_column_1].unique():
        for sensitive_2_i in data_frame[sensitive_column_2].unique():
            # Handle optional third sensitive column
            if sensitive_column_3:
                sensitive_3_values = data_frame[sensitive_column_3].unique()
            else:
                sensitive_3_values = [None]
                
            for sensitive_3_i in sensitive_3_values:
                for prob in data_frame[output_column].unique():
                    # Create subset filters
                    filters = (data_frame[sensitive_column_1] == sensitive_1_i) & \
                              (data_frame[sensitive_column_2] == sensitive_2_i) & \
                              (data_frame[output_column] == prob)
                    
                    # Add third column filter if it exists
                    if sensitive_column_3:
                        filters = filters & (data_frame[sensitive_column_3] == sensitive_3_i)
                        
                    # Create a subset for the specific group
                    subset = data_frame[filters]

                    # Create a file name
                    if sensitive_column_3:
                        file_name = f"{sensitive_column_1}_{sensitive_1_i}_{sensitive_column_2}_{sensitive_2_i}_{sensitive_column_3}_{sensitive_3_i}_{output_column}_{prob}.csv"
                    else:
                        file_name = f"{sensitive_column_1}_{sensitive_1_i}_{sensitive_column_2}_{sensitive_2_i}_{output_column}_{prob}.csv"

                    # Save the subset to a CSV file
                    if save_file:
                        subset.to_csv(os.path.join(folder_path, file_name), index=False)

                    # Create a dictionary with file info
                    file_info = {
                        'file_name': file_name,
                        'length': len(subset),
                        sensitive_column_1: sensitive_1_i,
                        sensitive_column_2: sensitive_2_i,
                        output_column: prob
                    }
                    
                    # Add third column info if it exists
                    if sensitive_column_3:
                        file_info[sensitive_column_3] = sensitive_3_i
                        
                    result_files.append(file_info)
                    print(f"Saved {file_name} with {len(subset)} rows")

    return result_files


# def get_desired_samples(infor_8_files):
#     split_table = [
#         {"Id": "D1", "sex": 1, "health": 1, "Probability": 1, "Cardinality": 0, "Expected_Cardinality": 0},
#         {"Id": "D2", "sex": 1, "health": 1, "Probability": 0, "Cardinality": 0, "Expected_Cardinality": 0},
#         {"Id": "D3", "sex": 1, "health": 0, "Probability": 1, "Cardinality": 0, "Expected_Cardinality": 0},
#         {"Id": "D4", "sex": 1, "health": 0, "Probability": 0, "Cardinality": 0, "Expected_Cardinality": 0},
#         {"Id": "D5", "sex": 0, "health": 1, "Probability": 1, "Cardinality": 0, "Expected_Cardinality": 0},
#         {"Id": "D6", "sex": 0, "health": 1, "Probability": 0, "Cardinality": 0, "Expected_Cardinality": 0},
#         {"Id": "D7", "sex": 0, "health": 0, "Probability": 1, "Cardinality": 0, "Expected_Cardinality": 0},
#         {"Id": "D8", "sex": 0, "health": 0, "Probability": 0, "Cardinality": 0, "Expected_Cardinality": 0}
#     ]
#
#     def get_sample_by_id(sample_id):
#         for index, split_i in enumerate(split_table):
#             if split_i["Id"] == sample_id:
#                 return index, split_i
#
#     def is_even_id(sample_id):
#         """
#
#         :param sample_id: D1, D2, D3, D4, D5, D6, D7, D8
#         :return: if sample_id is odd, return True, else return False
#         """
#         sample_index = int(sample_id[-1])
#         if sample_index % 2 == 0:
#             return True
#         return False
#
#     min_sample = None
#
#     for file_infor_i in infor_8_files:
#         sex_i, health_i, probability_i = file_infor_i["sex"], file_infor_i["health"], file_infor_i["Probability"]
#         for split_i in split_table:
#             if split_i["sex"] == sex_i and split_i["health"] == health_i and split_i["Probability"] == probability_i:
#                 split_i["Cardinality"] = file_infor_i["length"]
#                 if min_sample is None or split_i["Cardinality"] < min_sample["Cardinality"]:
#                     min_sample = split_i
#
#     D1_D2 = get_sample_by_id(min_sample["D1"])[1]["Cardinality"] / get_sample_by_id(min_sample["D2"])[1]["Cardinality"]
#     D3_D4 = get_sample_by_id(min_sample["D3"])[1]["Cardinality"] / get_sample_by_id(min_sample["D4"])[1]["Cardinality"]
#     D5_D6 = get_sample_by_id(min_sample["D5"])[1]["Cardinality"] / get_sample_by_id(min_sample["D6"])[1]["Cardinality"]
#     D7_D8 = get_sample_by_id(min_sample["D7"])[1]["Cardinality"] / get_sample_by_id(min_sample["D8"])[1]["Cardinality"]
#
#     R_tb = np.average([D1_D2, D3_D4, D5_D6, D7_D8])
#
#     base_D = min_sample["Cardinality"]
#     choose_base_D_new = base_D * 2
#     min_sample['Expected_Cardinality'] = choose_base_D_new
#
#     if is_even_id(min_sample['Id']):
#         for i in range(min_sample['Expected_Cardinality']):


if __name__ == '__main__':
    dataset_path = "student_performance.csv"  # input the original dataset path here
    path_to_save_training_data = 'student_performance_train.csv'
    path_to_save_test_data = 'student_performance_test.csv'

    folder_path_to_save_8_files = ""  # input path to save 8 files

    data_frame = pd.read_csv(dataset_path)
    data_frame = data_frame.drop(['school', 'address', 'famsize', 'Pstatus', 'Mjob', 'Fjob', 'reason', 'guardian'],
                                 axis=1)
    data_frame = data_frame.dropna()

    data_frame['sex'] = np.where(data_frame['sex'] == 'M', 1, 0)
    data_frame['health'] = np.where(data_frame['health'] >= 4, 1, 0)

    ## Make goal column binary
    mean = data_frame.loc[:, "Probability"].mean()
    data_frame['Probability'] = np.where(data_frame['Probability'] >= mean, 1, 0)

    train_df, test_df = split_train_test_csv_file(data_frame, train_file='student_performance_train.csv',
                                                  test_file='student_performance_test.csv')

    infor_8_files = split_into_8_files_student_performance(data_frame=train_df,
                                                           folder_path=folder_path_to_save_8_files)

    # chỗ này làm tay
