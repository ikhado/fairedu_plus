import argparse
import os
import numpy as np

from fairedu import run_fairedu
from generate_sdg import CONSTANT_CTGAN, CONSTANT_LLM, prepare_generate_sdg


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate synthetic data and run FaireduPlus on a chosen dataset."
    )
    parser.add_argument(
        "--dataset-name",
        default="student_dropout",
        choices=["student_dropout", "student_oulad", "student_performance", "DNU"],
        help="Dataset to process.",
    )
    parser.add_argument(
        "--dataset-folder",
        default=None,
        help="Root folder containing the dataset splits.",
    )
    parser.add_argument(
        "--generator",
        default=CONSTANT_LLM,
        type=str.upper,
        choices=[CONSTANT_LLM, CONSTANT_CTGAN],
        help="Synthetic data generator to use.",
    )
    parser.add_argument(
        "--merged-output-file-name",
        default="merged_output.csv",
        help="Filename for the merged generated output.",
    )
    parser.add_argument(
        "--run-splitted-file",
        dest="run_splitted_file",
        action="store_true",
        default=True,
        help="Generate data for each split file.",
    )
    parser.add_argument(
        "--no-run-splitted-file",
        dest="run_splitted_file",
        action="store_false",
        help="Generate data using the non-split file.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    np.random.seed(args.seed)

    test_dataset_path = os.path.join(
        args.dataset_folder, args.dataset_name, f"test_{args.dataset_name}.csv"
    )

    print("==================RUNNING==================")
    train_dataset_path = prepare_generate_sdg(
        args.dataset_folder,
        args.dataset_name,
        args.generator,
        output_file_name=args.merged_output_file_name,
        is_run_splitt_file=args.run_splitted_file,
    )
    run_fairedu(args.dataset_name, train_dataset_path, test_dataset_path)
    print("==================DONE=====================")


if __name__ == "__main__":
    main()
