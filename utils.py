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


new_DNU_COLUMNS = [
    "Điểm môn Giải tích 1",
    "Điểm môn Kỹ năng mềm",
    "Điểm môn Pháp luật đại cương",
    "Điểm môn Tiếng Anh 1",
    "Điểm môn Tin học đại cương",
    "Điểm môn Đại số tuyến tính",
    "Điểm môn Nhập môn lập trình",
    "Điểm môn Giải tích 2",
    "Điểm môn Mạng máy tính",
    "Điểm môn Những nguyên lý cơ bản của chủ nghĩa Marx-Lenin 1",
    "Điểm môn Tiếng Anh 2",
    "Điểm môn Cấu trúc dữ liệu và giải thuật",
    "Điểm môn Hệ quản trị cơ sở dữ liệu",
    "Điểm môn Những nguyên lý cơ bản của chủ nghĩa Marx-Lenin 2",
    "Điểm môn Quản trị mạng",
    "Điểm môn Tiếng Anh 3",
    "Điểm môn Xác suất và thống kê",
    "Điểm môn Hệ điều hành mã nguồn mở",
    "Điểm môn Lập trình hướng đối tượng",
    "Điểm môn Lý thuyết cơ sở dữ liệu",
    "Điểm môn Tiếng Anh 4",
    "Điểm môn Toán rời rạc",
    "Điểm môn Tư tưởng Hồ Chí Minh",
    "Điểm môn Cài đặt và bảo trì hệ thống máy tính",
    "Điểm môn Đường lối cách mạng của Đảng Cộng sản Việt Nam",
    "Điểm môn Cơ sở dữ liệu phân tán",
    "Điểm môn Lập trình .NET cơ bản",
    "Điểm môn Quản lý dự án CNTT",
    "Điểm môn Tiếng Anh chuyên ngành CNTT",
    "Điểm môn An toàn dữ liệu",
    "Điểm môn Lập trình Java",
    "Điểm môn Phân tích và thiết kế hệ thống",
    "Điểm môn TOEIC",
    "Probability",
]
