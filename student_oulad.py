import os

import numpy as np
import pandas as pd

from split_dataset import split_into_8_files, split_train_test_csv_file

origin_dataset_path = 'original_dataset'
current_dataset_name = 'student_oulad'

current_dataset_folder = os.path.join(origin_dataset_path, current_dataset_name)

if __name__ == '__main__':
    df = pd.read_csv(os.path.join(current_dataset_folder, current_dataset_name + '.csv'))

    df = df.dropna(axis=0, how='any')
    df = df.drop(columns=['id_student'])

    # Change symbolics to numerics
    df['gender'] = np.where(df['gender'] == 'F', 0, 1)
    df['disability'] = np.where(df['disability'] == 'Y', 0, 1)

    df = df.drop(
        columns=['num_of_prev_attempts', 'code_module', 'code_presentation', 'highest_education'])

    df['final_result'] = df['final_result'].apply(lambda x: 0 if x in ['Fail', 'Withdrawn'] else 1)
    df = df.rename(columns={'final_result': 'Probability'})

    split_into_8_files(folder_path=current_dataset_folder, data_frame=df, sensitive_column_1='gender',
                       sensitive_column_2='disability', output_column='Probability', save_file=False)

    train_file, test_file = os.path.join(origin_dataset_path, 'train_' + current_dataset_name + '.csv'), os.path.join(
        origin_dataset_path, 'test_' + current_dataset_name + '.csv')

    training_set, test_set = split_train_test_csv_file(data_frame=df, train_file=train_file, test_file=test_file)

    split_into_8_files(folder_path=current_dataset_folder, data_frame=training_set, sensitive_column_1='gender',
                       sensitive_column_2='disability', output_column='Probability', save_file=True)
