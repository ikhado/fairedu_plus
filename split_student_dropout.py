import os

import numpy as np
import pandas as pd

from split_dataset import split_into_k_files, split_train_test_csv_file

origin_dataset_path = 'original_dataset'
current_dataset_name = 'student_dropout'

current_dataset_folder = os.path.join(origin_dataset_path, current_dataset_name)

if __name__ == '__main__':
    df = pd.read_csv(os.path.join(current_dataset_folder, current_dataset_name + '.csv'))

    df['Probability'] = np.where(df['Target'] == 'Graduate', 1, 0)
    df = df.drop(['Target'], axis=1)

    train_file, test_file = os.path.join(origin_dataset_path, 'train_' + current_dataset_name + '.csv'), os.path.join(
        origin_dataset_path, 'test_' + current_dataset_name + '.csv')

    training_set, test_set = split_train_test_csv_file(data_frame=df, train_file=train_file, test_file=test_file)

    split_into_k_files(folder_path=current_dataset_folder, data_frame=training_set, sensitive_column_1='Gender',
                       sensitive_column_2='Debtor', output_column='Probability', save_file=True)