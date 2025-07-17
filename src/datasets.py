import argparse
import datetime
from math import floor
import os
import random
import numpy as np
import pandas as pd
from pydantic import BaseModel
import torch


class TrajectoryDataset(torch.utils.data.Dataset):
    """
    Dataset that is built from a list of paths to .csv files. 
    Currently assumes that you have velocity data.
    """
    def __init__(self, files: list[str], input_length: int, output_length: int):
        self.input_length = input_length
        self.output_length = output_length
        self.total_sequence_length = input_length + output_length

        self.sample_map: list[tuple[int, int, int]] = []
        
        self.data_arrays: list[np.ndarray] = [] 
        
        current_global_index = 0

        for df_idx, file in enumerate(files):
            try:
                df: pd.DataFrame = pd.read_csv(file, usecols=["vx", "vy", "vz"])
            except Exception as e:
                print(f"Error reading {file}: {e}. Skipping.")
                continue

            data_array = df.values.astype(np.float32) # Ensure float32 here
            
            if len(data_array) < self.total_sequence_length:
                print(f"{file} is too short ({len(data_array)} rows) for input_length={input_length} and output_length={output_length}. Skipping.")
                continue
            
            num_sequences_in_df = len(data_array) - self.total_sequence_length + 1
            
            for i in range(num_sequences_in_df):
                self.sample_map.append((current_global_index + i, df_idx, i))
            
            current_global_index += num_sequences_in_df
            self.data_arrays.append(data_array) # Store the NumPy array

        self.total_samples = current_global_index

    def __len__(self) -> int:
        return self.total_samples

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        if not (0 <= index < self.total_samples):
            raise IndexError(f"Index {index} is out of bounds for dataset of size {self.total_samples}")

        global_index, df_idx, local_start_row = self.sample_map[index]

        data_array = self.data_arrays[df_idx] # Retrieve NumPy array

        x_start_local = local_start_row
        x_end_local = local_start_row + self.input_length

        y_start_local = x_end_local
        y_end_local = y_start_local + self.output_length
        
        # Slice NumPy arrays (very fast)
        x_data = data_array[x_start_local:x_end_local]
        y_data = data_array[y_start_local:y_end_local]
        
        # Convert slices to PyTorch tensors (still happens in __getitem__, but from NumPy)
        # This conversion is very efficient from NumPy arrays
        x_tensor = torch.from_numpy(x_data) 
        y_tensor = torch.from_numpy(y_data)
        
        return x_tensor, y_tensor
    

class Fold(BaseModel):
    train: list[str]
    validation: list[str]

class Split(BaseModel):
    test: list[str]
    folds: list[Fold]

def abs_path(root: str, stratum: str, file: str):
    return os.path.abspath(os.path.join(root, stratum, file))

def generate_split(
    root: str, k: int = 5, shuffle: bool = False, seed: int = 1
) -> Split :
    strata = os.listdir(root)

    folds: list[Fold] = [
        Fold(train=[], validation=[])
        for _ in range(k)
    ]
    split = Split(test=[], folds=folds)

    # Assume all csv's have unique names
    for stratum in strata:
        files = os.listdir(os.path.join(root, stratum))
        if shuffle:
            random.seed(seed)
            random.shuffle(files)

        m = len(files)
        
        # Global test set
        test_size = floor(m * 0.10)
        if test_size == 0: 
            raise Exception("Dataset too small, not enough for test set")

        split.test.extend(abs_path(root, stratum, x) for x in files[0:test_size])

        fold_set = files[test_size:]
        val_size = len(fold_set) // k
        if test_size == 0: 
            raise Exception("Dataset too small, not enough for validation set")
        for i in range(k):
            val_start = i * val_size
            val_end = (i + 1) * val_size

            folds[i].train.extend(abs_path(root, stratum, x) for x in fold_set[:val_start])
            folds[i].validation.extend(abs_path(root, stratum, x) for x in fold_set[val_start:val_end])
            folds[i].train.extend(abs_path(root, stratum, x) for x in fold_set[val_end:])

    return split

def save_split(split: Split, output_file: str | None = None):
    if not output_file:
        current_time = datetime.datetime.now()
        timestamp_str = current_time.strftime("%Y%m%d_%H%M%S_%f")[:-3]
        output_file = f"data/data_split_{timestamp_str}.json"

    with open(output_file, "w", encoding="utf-8") as f:
        f.write(split.model_dump_json(indent=2))

def read_split(file_path: str) -> Split:
    with open(file_path, 'r', encoding='utf-8') as f:
        raw_data = f.read()
    
    try:
        return Split.model_validate_json(raw_data)

    except Exception as e:
        raise e

def main():
    parser = argparse.ArgumentParser(
        description="Generate a data split for machine learning experiments."
    )

    parser.add_argument(
        "root",
        type=str,
        help="The root directory path for the dataset.",
    )

    parser.add_argument(
        "-k",
        "--k-folds",
        type=int,
        default=5,  # Matches the function's default
        help="The number of folds/splits to generate (default: 5).",
    )

    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Enable shuffling of data before splitting. (Default: NO shuffle)",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=1, 
        help="The random seed for reproducibility (default: 1).",
    )

    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help="Optional: Path to save the generated split. "
             "If not provided, a default timestamped file in 'data/' will be created.",
    )


    args = parser.parse_args()

    # Call the generate_split function with the parsed arguments
    # The attributes of 'args' directly correspond to the argument names defined.
    result_split = generate_split(
        root=args.root,
        k=args.k_folds,  # Use args.k_folds because we named it '--k-folds'
        shuffle=args.shuffle,
        seed=args.seed,
    )
    
    if args.output: 
        save_split(result_split, args.output)
    else:
        print(result_split.model_dump_json(indent=2))

if __name__ == "__main__":
    main()