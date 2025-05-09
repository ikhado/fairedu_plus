import os

import numpy as np
import pandas as pd

from split_dataset import split_into_k_files, split_train_test_csv_file

origin_dataset_path = 'original_dataset'
current_dataset_name = 'DNU'

current_dataset_folder = os.path.join(origin_dataset_path, current_dataset_name)


def preprocess_student_DNU():
    data_path = '/home/ad/m4do/proj/fairedu_plus/original_dataset/DNU/llm_no_8files/train_DNU_LLM.csv'
    df = pd.read_csv(data_path)

    # Change symbolics to numerics
    # df['gender'] = np.where(df['gender'] == 'Female', 0, 1)
    # df['disability'] = np.where(df['disability'] == 'Yes', 0, 1)
    ## Make goal column binary
    mean = df.loc[:, "Probability"].mean()
    df['Probability'] = np.where(df['Probability'] >= mean, 1, 0)
    df.to_csv(data_path, index=False)

if __name__ == '__main__':
    preprocess_student_DNU()
    # df = pd.read_csv(os.path.join(current_dataset_folder, current_dataset_name + '.csv'))
    #
    # df = df.drop(['adv_score'], axis=1)
    # # Change Column values
    # df['gender'] = np.where(df['gender'] == 'Male', 1, 0)
    #
    # train_file, test_file = os.path.join(origin_dataset_path, 'train_' + current_dataset_name + '.csv'), os.path.join(
    #     origin_dataset_path, 'test_' + current_dataset_name + '.csv')
    #
    # split_into_k_files(folder_path=current_dataset_folder, data_frame=df, sensitive_column_1='gender',
    #                    sensitive_column_2='age', sensitive_column_3='birthplace', output_column='Probability',
    #                    save_file=False)
    #
    # training_set, test_set = split_train_test_csv_file(data_frame=df, train_file=train_file, test_file=test_file)
    #
    # split_into_k_files(folder_path=current_dataset_folder, data_frame=training_set, sensitive_column_1='gender',
    #                    sensitive_column_2='age', sensitive_column_3='birthplace', output_column='Probability',
    #                    save_file=True)
