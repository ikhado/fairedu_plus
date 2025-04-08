import os

import pandas as pd

from ctgan import CTGAN

from utils import merge_csv_files

CONSTANT_CTGAN = 'CTGAN'
CONSTANT_LLM = 'LLM'


def generate_by_ctgan(real_data, num_rows):
    ctgan = CTGAN(epochs=10)
    discrete_columns = real_data.columns.tolist()

    ctgan.fit(real_data, discrete_columns)
    # Create synthetic data
    synthetic_data = ctgan.sample(num_rows)
    return synthetic_data


def generate_sdg_CTGAN(file_dict, path_8_files, generator=CONSTANT_LLM, sensitive_column_1='sex',
                       sensitive_column_2='health',
                       output_column='Probability'):
    for curr_file in file_dict:
        path = curr_file['file_name']
        expected_value = curr_file['expected_value']

        current_path = os.path.join(path_8_files, path)
        current_df = pd.read_csv(current_path)

        print(f"File: {path}")
        print(f"Length: {len(current_df)}")

        if 'ctgan_no_8files' not in path_8_files:
            sex_value = current_df[sensitive_column_1].unique()[0]
            health_value = current_df[sensitive_column_2].unique()[0]
            prob_value = current_df[output_column].unique()[0]
            current_df = current_df.drop(columns=[sensitive_column_1, sensitive_column_2, output_column])

        sampled_data = None
        if generator == CONSTANT_CTGAN:
            sampled_data = generate_by_ctgan(real_data=current_df, num_rows=expected_value - len(current_df))

        if sampled_data is None:
            continue
        if 'ctgan_no_8files' not in path_8_files:
            sampled_data[sensitive_column_1] = sex_value
            sampled_data[sensitive_column_2] = health_value
            sampled_data[output_column] = prob_value

        save_csv_path = current_path.replace(".csv", "_" + generator + ".csv")
        sampled_data.to_csv(save_csv_path, index=False)


if __name__ == "__main__":
    dataset_name = 'student_performance'  # student_dropout, student_oulad student_performance
    original_dataset_path = './original_dataset'
    generated_dataset_path = './generated_dataset'

    path_to_8_files = os.path.join(original_dataset_path, dataset_name, '8_files')
    path_to_no_8_files = os.path.join(original_dataset_path, dataset_name, 'ctgan_no_8files')
    is_run_splitt_file = False

    sensitive_columns = ['sex', 'health']
    output_column = 'Probability'

    student_performance_path_dict = [
        {"file_name": "sex_1_health_1_prob_1.csv", "expected_value": 208},
        {"file_name": "sex_1_health_1_prob_0.csv", "expected_value": 229},
        {"file_name": "sex_1_health_0_prob_1.csv", "expected_value": 136},
        {"file_name": "sex_1_health_0_prob_0.csv", "expected_value": 150},
        {"file_name": "sex_0_health_1_prob_1.csv", "expected_value": 240},
        {"file_name": "sex_0_health_1_prob_0.csv", "expected_value": 264},
        {"file_name": "sex_0_health_0_prob_1.csv", "expected_value": 264},
        {"file_name": "sex_0_health_0_prob_0.csv", "expected_value": 291},
    ]

    if not is_run_splitt_file:
        student_performance_path_dict = [
            {"file_name": "train_student_performance.csv",
             "expected_value": sum(item["expected_value"] for item in student_performance_path_dict)},
        ]

    data_infor_dict = student_performance_path_dict

    if dataset_name == 'student_oulad':
        sensitive_columns = ['gender', 'disability']
        output_column = 'Probability'

        data_infor_dict = [
            {"file_name": "gender_1_disability_0_Probability_1.csv", "expected_value": 916},
            {"file_name": "gender_1_disability_0_Probability_0.csv", "expected_value": 1201},
            {"file_name": "gender_1_disability_1_Probability_1.csv", "expected_value": 12190},
            {"file_name": "gender_1_disability_1_Probability_0.csv", "expected_value": 15980},
            {"file_name": "gender_0_disability_0_Probability_1.csv", "expected_value": 1116},
            {"file_name": "gender_0_disability_0_Probability_0.csv", "expected_value": 1463},
            {"file_name": "gender_0_disability_1_Probability_1.csv", "expected_value": 10618},
            {"file_name": "gender_0_disability_1_Probability_0.csv", "expected_value": 13919},
        ]

        if not is_run_splitt_file:
            data_infor_dict = [
                {"file_name": "train_student_oulad.csv",
                 "expected_value": sum(item["expected_value"] for item in data_infor_dict)},
            ]

    if dataset_name == 'student_dropout':
        sensitive_columns = ['Gender', 'Debtor']
        output_column = 'Probability'

        data_infor_dict = [
            {"file_name": "Gender_1_Debtor_0_Probability_1.csv", "expected_value": 526},
            {"file_name": "Gender_1_Debtor_0_Probability_0.csv", "expected_value": 817},
            {"file_name": "Gender_1_Debtor_1_Probability_1.csv", "expected_value": 22},
            {"file_name": "Gender_1_Debtor_1_Probability_0.csv", "expected_value": 191},
            {"file_name": "Gender_0_Debtor_0_Probability_1.csv", "expected_value": 1582},
            {"file_name": "Gender_0_Debtor_0_Probability_0.csv", "expected_value": 996},
            {"file_name": "Gender_0_Debtor_1_Probability_1.csv", "expected_value": 79},
            {"file_name": "Gender_0_Debtor_1_Probability_0.csv", "expected_value": 211},
        ]

        if not is_run_splitt_file:
            data_infor_dict = [
                {"file_name": "train_student_dropout.csv",
                 "expected_value": sum(item["expected_value"] for item in data_infor_dict)},
            ]


    if is_run_splitt_file:
        generate_sdg_CTGAN(file_dict=data_infor_dict, path_8_files=path_to_8_files, generator=CONSTANT_CTGAN,
                           sensitive_column_1=sensitive_columns[0], sensitive_column_2=sensitive_columns[1],
                           output_column=output_column)

        # merge all files within path_to_8_files folder into "merged_output.csv" file
        merge_csv_files(folder_path=path_to_8_files, output_file="merged_output.csv")
    else:
        generate_sdg_CTGAN(file_dict=data_infor_dict, path_8_files=path_to_no_8_files, generator=CONSTANT_CTGAN,
                           sensitive_column_1=sensitive_columns[0], sensitive_column_2=sensitive_columns[1],
                           output_column=output_column)
        # merge all files within path_to_8_files folder into "merged_output.csv" file
        merge_csv_files(folder_path=path_to_no_8_files, output_file="merged_output.csv")