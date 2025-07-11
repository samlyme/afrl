import json
from math import floor
import os
import random
from typing import TypedDict
import numpy as np
import pandas as pd
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
    

class Fold(TypedDict):
    train: list[str]
    validation: list[str]
    test: list[str]


def generate_folds(
    root: str, strata: list[str], k: int = 5, shuffle: bool = False,
) -> list[Fold] :
    folds: list[Fold] = [
        {"train": [], "validation": [], "test": []}
        for _ in range(k)
    ]
    # Assume all csv's have unique names
    for stratum in strata:
        files = os.listdir(os.path.join(root, stratum))
        if shuffle:
            random.shuffle(files)

        m = len(files)
        
        for i in range(k):
            fold_start = floor(m * (i/k))
            fold_end = floor(m * ((i+1)/k))
            
            fold = files[fold_start:fold_end]

            fold_train = floor(len(fold) * 0.65)
            fold_validation = floor(len(fold) * 0.85)

            folds[i]["train"].extend(
                [os.path.realpath(os.path.join(root, stratum, f)) for f in fold[0 : fold_train]]
            )
            folds[i]["validation"].extend(
                [os.path.realpath(os.path.join(root, stratum, f)) for f in fold[fold_train : fold_validation]]
            )
            folds[i]["test"].extend(
                [os.path.realpath(os.path.join(root, stratum, f)) for  f in fold[fold_validation :]]
            )
    return folds

def save_folds(folds: list[Fold], output_file: str):
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(folds, f, indent=4)