import argparse
import numpy as np
import pandas as pd
import torch

def resample(df: pd.DataFrame, sampling_time: float, coords = ["x", "y", "z"]) -> pd.DataFrame:
    """
    This function takes the block average as a simple way to reduce noise
    """
    df = df.sort_values("timestamp")
    bins = np.arange(
        df['timestamp'].min(), 
        df['timestamp'].max() + sampling_time + 1e-9, 
        sampling_time
    )
    
    df["bin_index"] = np.digitize(df["timestamp"], bins, right=True)

    grouped = df.groupby("bin_index")
    mean_values = grouped[coords].mean()

    unique_grouped_indices = mean_values.index.values

    valid_indices_mask = unique_grouped_indices > 0
    
    filtered_bin_indices = unique_grouped_indices[valid_indices_mask]
    filtered_mean_values = mean_values[valid_indices_mask]

    df_resampled = pd.DataFrame({
        # For a bin_index 'k', the block starts at bins[k-1]
        'timestamp': bins[filtered_bin_indices - 1],
        coords[0]: filtered_mean_values[coords[0]].values,
        coords[1]: filtered_mean_values[coords[1]].values,
        coords[2]: filtered_mean_values[coords[2]].values,
    })
    
    return df_resampled


class FlightDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        csv_path: str,
        original_sr: float = 500.0,
        target_sr: float = 100.0,
        window_size: int = 20,
        norm_out: bool = False, # if True, getitem will always have the first value be (0, 0, 0)
    ):
        df = pd.read_csv(csv_path).sort_values("timestamp")
        data_np = df[["x", "y", "z"]].to_numpy(dtype="float32")
        self.data = torch.from_numpy(data_np)           # (N,3)

        self.norm_out = norm_out

        # 2) Compute integer decimation factor
        if target_sr > original_sr:
            raise ValueError("target_sr must be <= original_sr")
        factor = int(original_sr / target_sr)
        if original_sr % target_sr != 0:
            raise ValueError("original_sr must be a multiple of target_sr")
        self.factor = factor

        # 3) Sliding window params
        self.window_size = window_size
        self.win_orig = window_size * factor # raw points per window
        N = self.data.size(0)
        self.n_windows = N - self.win_orig + 1
        if self.n_windows < 1:
            raise ValueError("window_size too large for data length")


    def __len__(self):
        return self.n_windows

    def __getitem__(self, idx: int):
        # slice raw 500 Hz segment
        start = idx
        end   = idx + self.win_orig
        # strided slice: decimation + window
        window = self.data[start:end:self.factor, :]     # (window_size,3)
        if self.norm_out:
            window = window - window[0]

        return window


class FlightsDataset(torch.utils.data.ConcatDataset):
    def __init__(self, flights: list[FlightDataset]):
        super().__init__(flights)

def generate_auto_encoder_dataset(
    files: list[str],
    input_sr: float, 
    target_sr: float, 
    window_size: int,
    norm_out: bool = False
):
    return FlightsDataset(
        flights=[
            FlightDataset(file, input_sr, target_sr, window_size, norm_out) 
            for file in files if file.endswith(".csv")
        ]
    )

def parse_args():
    p = argparse.ArgumentParser(
        description="Build an AutoEncoderDataset from one or more CSV files"
    )
    p.add_argument(
        "files",
        nargs="+",
        help="Input CSV files (must end with .csv)",
    )
    p.add_argument(
        "--input-sr",
        type=float,
        required=True,
        help="Original sample rate (Hz)",
    )
    p.add_argument(
        "--target-sr",
        type=float,
        required=True,
        help="Target (downsampled) sample rate (Hz)",
    )
    p.add_argument(
        "--window-size",
        type=int,
        required=True,
        help="Number of downsampled samples per window",
    )
    p.add_argument(
        "--norm-out",
        action="store_true",
        help="Subtract first sample of each window (zero-offset)",
    )
    p.add_argument(
        "--save-to",
        default="ae_dataset.pt",
        help="Path to save the resulting dataset (torch.save)",
    )
    return p.parse_args()

def main():
    args = parse_args()

    ds = generate_auto_encoder_dataset(
        files=args.files,
        input_sr=args.input_sr,
        target_sr=args.target_sr,
        window_size=args.window_size,
        norm_out=args.norm_out,
    )

    print(f"Built dataset with {len(ds)} windows.")
    torch.save(ds, args.save_to)
    print(f"Saved dataset to '{args.save_to}'")

if __name__ == "__main__":
    main()