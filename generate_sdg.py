import os

import pandas as pd

from ctgan import CTGAN

from utils import merge_csv_files, new_DNU_COLUMNS
from sdgx.utils import download_demo_data
from sdgx.data_models.metadata import Metadata
from sdgx.models.LLM.single_table.gpt import SingleTableGPTModel
import numpy as np

CONSTANT_CTGAN = 'CTGAN'
CONSTANT_LLM = 'LLM'


def preprocess_student_performance(data_frame):
    data_frame['sex'] = np.where(data_frame['sex'] == 'M', 1, 0)
    data_frame['health'] = np.where(data_frame['health'] >= 4, 1, 0)

    ## Make goal column binary
    mean = data_frame.loc[:, "Probability"].mean()
    data_frame['Probability'] = np.where(data_frame['Probability'] >= mean, 1, 0)
    return data_frame


def preprocess_student_oulad(data_frame):
    # todo provide preprocess_student_oulad here
    return data_frame


def preprocess_student_dropout(data_frame):
    # todo provide preprocess_student_oulad here
    return data_frame


def generate_by_ctgan(real_data, num_rows):
    ctgan = CTGAN(epochs=10)
    discrete_columns = real_data.columns.tolist()

    ctgan.fit(real_data, discrete_columns)
    # Create synthetic data
    synthetic_data = ctgan.sample(num_rows)
    return synthetic_data


import pandas as pd
from typing import Iterable, Optional


def generate_by_llm(real_data, num_rows):
    if num_rows <= 0:
        return None
    OPEN_AI_BASE = "https://api.openai.com/v1/"
    OPEN_AI_KEY = os.getenv('OPENAI_API_KEY')
    # real_data = multiply_columns_by_10(csv_in=real_data, csv_out=None)

    metadata = Metadata.from_dataframe(real_data)

    model = SingleTableGPTModel()
    model.set_openAI_settings(OPEN_AI_BASE, OPEN_AI_KEY)
    model.gpt_model = "gpt-3.5-turbo"
    model.fit(metadata)

    sampled_data = model.sample(num_rows)
    return sampled_data


def generate_sdg(file_dict, path_8_files, generator=CONSTANT_LLM, sensitive_column_1='sex',
                 sensitive_column_2='health', sensitive_column_3=None,
                 output_column='Probability', is_run_splitt_file=True):
    for curr_file in file_dict:
        path = curr_file['file_name']
        expected_value = curr_file['expected_value']
        if expected_value == 0:
            continue

        current_path = os.path.join(path_8_files, path)
        current_df = pd.read_csv(current_path, encoding='utf-8')

        print(f"File: {path}")
        print(f"Length: {len(current_df)}")

        if is_run_splitt_file:
            sex_value = current_df[sensitive_column_1].unique()[0]
            health_value = current_df[sensitive_column_2].unique()[0]
            prob_value = current_df[output_column].unique()[0]
            current_df = current_df.drop(columns=[sensitive_column_1, sensitive_column_2, output_column])

        sampled_data = None
        if generator == CONSTANT_CTGAN:
            sampled_data = generate_by_ctgan(real_data=current_df, num_rows=expected_value - len(current_df))
        elif generator == CONSTANT_LLM:
            current_columns = current_df.columns.tolist()
            if 'dnu' in path_8_files.lower():
                current_df.columns = new_DNU_COLUMNS
            sampled_data = generate_by_llm(real_data=current_df, num_rows=expected_value - len(current_df))
            if sampled_data is not None:
                sampled_data.columns = current_columns

        if sampled_data is None:
            continue
        if is_run_splitt_file:
            sampled_data[sensitive_column_1] = sex_value
            sampled_data[sensitive_column_2] = health_value
            sampled_data[output_column] = prob_value

        save_csv_path = current_path.replace(".csv", "_" + generator + ".csv")
        sampled_data.to_csv(save_csv_path, index=False)


if __name__ == "__main__":
    dataset_name = 'student_performance'  # student_dropout, student_oulad student_performance, DNU
    original_dataset_path = './original_dataset'
    generated_dataset_path = './generated_dataset'
    generator = CONSTANT_CTGAN

    if generator == CONSTANT_LLM:
        path_to_8_files = os.path.join(original_dataset_path, dataset_name, 'llm_8_files')
        path_to_no_8_files = os.path.join(original_dataset_path, dataset_name, 'llm_no_8files')
    elif generator == CONSTANT_CTGAN:
        path_to_8_files = os.path.join(original_dataset_path, dataset_name, '8_files')
        path_to_no_8_files = os.path.join(original_dataset_path, dataset_name, 'ctgan_no_8files')

    is_run_splitt_file = True

    sensitive_columns = ['sex', 'health', None]
    output_column = 'Probability'

    student_performance_path_dict = [
        {"file_name": "sex_1_health_1_prob_1.csv", "expected_value": 153},
        {"file_name": "sex_1_health_1_prob_0.csv", "expected_value": 168},
        {"file_name": "sex_1_health_0_prob_1.csv", "expected_value": 100},
        {"file_name": "sex_1_health_0_prob_0.csv", "expected_value": 110},
        {"file_name": "sex_0_health_1_prob_1.csv", "expected_value": 176},
        {"file_name": "sex_0_health_1_prob_0.csv", "expected_value": 194},
        {"file_name": "sex_0_health_0_prob_1.csv", "expected_value": 194},
        {"file_name": "sex_0_health_0_prob_0.csv", "expected_value": 214},
    ]

    if not is_run_splitt_file:
        student_performance_path_dict = [
            {"file_name": "train_student_performance.csv",
             "expected_value": sum(item["expected_value"] for item in student_performance_path_dict)},
        ]

    data_infor_dict = student_performance_path_dict

    if dataset_name == 'student_oulad':
        sensitive_columns = ['gender', 'disability', None]
        output_column = 'Probability'

        data_infor_dict = [
            {"file_name": "gender_1_disability_0_Probability_1.csv", "expected_value": 656},
            {"file_name": "gender_1_disability_0_Probability_0.csv", "expected_value": 860},
            {"file_name": "gender_1_disability_1_Probability_1.csv", "expected_value": 8731},
            {"file_name": "gender_1_disability_1_Probability_0.csv", "expected_value": 11445},
            {"file_name": "gender_0_disability_0_Probability_1.csv", "expected_value": 799},
            {"file_name": "gender_0_disability_0_Probability_0.csv", "expected_value": 1048},
            {"file_name": "gender_0_disability_1_Probability_1.csv", "expected_value": 7605},
            {"file_name": "gender_0_disability_1_Probability_0.csv", "expected_value": 9969},
        ]

        if not is_run_splitt_file:
            data_infor_dict = [
                {"file_name": "train_student_oulad.csv",
                 "expected_value": sum(item["expected_value"] for item in data_infor_dict)},
            ]

    if dataset_name == 'student_dropout':
        sensitive_columns = ['Gender', 'Debtor', None]
        output_column = 'Probability'

        data_infor_dict = [
            {"file_name": "Gender_1_Debtor_0_Probability_1.csv", "expected_value": 2440},
            {"file_name": "Gender_1_Debtor_0_Probability_0.csv", "expected_value": 3592},
            {"file_name": "Gender_1_Debtor_1_Probability_1.csv", "expected_value": 111},
            {"file_name": "Gender_1_Debtor_1_Probability_0.csv", "expected_value": 164},
            {"file_name": "Gender_0_Debtor_0_Probability_1.csv", "expected_value": 7453},
            {"file_name": "Gender_0_Debtor_0_Probability_0.csv", "expected_value": 10972},
            {"file_name": "Gender_0_Debtor_1_Probability_1.csv", "expected_value": 395},
            {"file_name": "Gender_0_Debtor_1_Probability_0.csv", "expected_value": 582},
        ]

        if not is_run_splitt_file:
            data_infor_dict = [
                {"file_name": "train_student_dropout.csv",
                 "expected_value": sum(item["expected_value"] for item in data_infor_dict)},
            ]

    if dataset_name == 'DNU':
        sensitive_columns = ['gender', 'age', 'birthplace']
        output_column = 'Probability'

        data_infor_dict = [
            {"file_name": "gender_0_age_0_birthplace_0_Probability_0.csv", "expected_value": 0},
            {"file_name": "gender_0_age_0_birthplace_0_Probability_1.csv", "expected_value": 23},
            {"file_name": "gender_0_age_0_birthplace_1_Probability_0.csv", "expected_value": 0},
            {"file_name": "gender_0_age_0_birthplace_1_Probability_1.csv", "expected_value": 5},
            {"file_name": "gender_0_age_1_birthplace_0_Probability_0.csv", "expected_value": 5},
            {"file_name": "gender_0_age_1_birthplace_0_Probability_1.csv", "expected_value": 54},
            {"file_name": "gender_0_age_1_birthplace_1_Probability_0.csv", "expected_value": 5},
            {"file_name": "gender_0_age_1_birthplace_1_Probability_1.csv", "expected_value": 54},

            {"file_name": "gender_1_age_0_birthplace_0_Probability_0.csv", "expected_value": 53},
            {"file_name": "gender_1_age_0_birthplace_0_Probability_1.csv", "expected_value": 572},
            {"file_name": "gender_1_age_0_birthplace_1_Probability_0.csv", "expected_value": 8},
            {"file_name": "gender_1_age_0_birthplace_1_Probability_1.csv", "expected_value": 86},
            {"file_name": "gender_1_age_1_birthplace_0_Probability_0.csv", "expected_value": 38},
            {"file_name": "gender_1_age_1_birthplace_0_Probability_1.csv", "expected_value": 410},
            {"file_name": "gender_1_age_1_birthplace_1_Probability_0.csv", "expected_value": 20},
            {"file_name": "gender_1_age_1_birthplace_1_Probability_1.csv", "expected_value": 216},
        ]

        if not is_run_splitt_file:
            data_infor_dict = [
                {"file_name": "train_DNU.csv",
                 "expected_value": sum(item["expected_value"] for item in data_infor_dict)},
            ]

    if is_run_splitt_file:
        generate_sdg(file_dict=data_infor_dict, path_8_files=path_to_8_files, generator=generator,
                     sensitive_column_1=sensitive_columns[0], sensitive_column_2=sensitive_columns[1],
                     output_column=output_column, is_run_splitt_file=is_run_splitt_file)

        # merge all files within path_to_8_files folder into "merged_output.csv" file
        # merge_csv_files(folder_path=path_to_8_files, output_file="merged_output.csv")
    else:
        generate_sdg(file_dict=data_infor_dict, path_8_files=path_to_no_8_files, generator=generator,
                     sensitive_column_1=sensitive_columns[0], sensitive_column_2=sensitive_columns[1],
                     output_column=output_column, is_run_splitt_file=is_run_splitt_file, sensitive_column_3=sensitive_columns[2])
        # merge all files within path_to_8_files folder into "merged_output.csv" file
        # merge_csv_files(folder_path=path_to_no_8_files, output_file="merged_output.csv")
