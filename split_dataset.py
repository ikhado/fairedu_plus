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


def preprocess_student_oulad():
    data_path = '/home/ad/m4do/proj/fairedu_plus/original_dataset/student_oulad/llm_no_8files/train_student_oulad_LLM.csv'
    df = pd.read_csv(data_path)

    # Change symbolics to numerics
    df['gender'] = np.where(df['gender'] == 'Female', 0, 1)
    df['disability'] = np.where(df['disability'] == 'Yes', 0, 1)
    ## Make goal column binary
    mean = df.loc[:, "Probability"].mean()
    df['Probability'] = np.where(df['Probability'] >= mean, 1, 0)
    df.to_csv(data_path, index=False)


if __name__ == '__main__':
    # dataset_path = "student_performance.csv"  # input the original dataset path here
    # path_to_save_training_data = 'student_performance_train.csv'
    # path_to_save_test_data = 'student_performance_test.csv'
    #
    # folder_path_to_save_8_files = ""  # input path to save 8 files
    #
    # data_frame = pd.read_csv(dataset_path)
    # data_frame = data_frame.drop(['school', 'address', 'famsize', 'Pstatus', 'Mjob', 'Fjob', 'reason', 'guardian'],
    #                              axis=1)
    # data_frame = data_frame.dropna()
    #
    # data_frame['sex'] = np.where(data_frame['sex'] == 'M', 1, 0)
    # data_frame['health'] = np.where(data_frame['health'] >= 4, 1, 0)
    #
    # ## Make goal column binary
    # mean = data_frame.loc[:, "Probability"].mean()
    # data_frame['Probability'] = np.where(data_frame['Probability'] >= mean, 1, 0)
    #
    # train_df, test_df = split_train_test_csv_file(data_frame, train_file='student_performance_train.csv',
    #                                               test_file='student_performance_test.csv')
    #
    # infor_8_files = split_into_8_files_student_performance(data_frame=train_df,
    #                                                        folder_path=folder_path_to_save_8_files)

    preprocess_student_oulad()
