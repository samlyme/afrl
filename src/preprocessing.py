
import argparse
import os
import numpy as np
import pandas as pd


def resample(df: pd.DataFrame, sampling_time: float):
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
    mean_values = grouped[["tx", "ty", "tz"]].mean()

    unique_grouped_indices = mean_values.index.values

    valid_indices_mask = unique_grouped_indices > 0
    
    filtered_bin_indices = unique_grouped_indices[valid_indices_mask]
    filtered_mean_values = mean_values[valid_indices_mask]

    df_resampled = pd.DataFrame({
        # For a bin_index 'k', the block starts at bins[k-1]
        'timestamp': bins[filtered_bin_indices - 1],
        'tx': filtered_mean_values["tx"].values,
        'ty': filtered_mean_values["ty"].values,
        'tz': filtered_mean_values["tz"].values
    })
    
    return df_resampled

def pos_to_vel(df: pd.DataFrame):
    out = pd.DataFrame(columns=["timestamp", "vx", "vy", "vz"])
    dt = df["timestamp"].diff()

    out["timestamp"] = df["timestamp"]
    out["vx"] = df["tx"].diff() / dt
    out["vy"] = df["ty"].diff() / dt
    out["vz"] = df["tz"].diff() / dt

    return out.iloc[1:]

def vel_to_acc(df: pd.DataFrame):
    out = pd.DataFrame(columns=["timestamp", "ax", "ay", "az"])
    dt = df["timestamp"].diff()

    out["timestamp"] = df["timestamp"]
    out["ax"] = df["vx"].diff() / dt
    out["ay"] = df["vy"].diff() / dt
    out["az"] = df["vz"].diff() / dt

    return out.iloc[1:]


def walk_and_process(
    root: str, 
    out_path_pos: str, 
    out_path_vel: str, 
    out_path_acc: str
):
    os.makedirs(out_path_pos, exist_ok=True)
    os.makedirs(out_path_vel, exist_ok=True)
    os.makedirs(out_path_acc, exist_ok=True)
    coords = ["x", "y", "z"]
    for dirpath, _, filenames in os.walk(root):
        # Position is always resampled
        for filename in filenames:
            if "500hz_freq_sync.csv" in filename:
                df: pd.DataFrame = pd.read_csv(os.path.join(dirpath, filename))

                # From the racing dataset
                pos: pd.DataFrame = df[["elapsed_time"] + ["drone_" + d for d in coords]]
                pos.rename({"drone_" + d: d for d in coords})
                pos.to_csv(os.path.join(out_path_pos, filename))

                vel: pd.DataFrame = df[["elapsed_time"] + ["drone_velocity_linear_" + d for d in coords]]
                vel.rename({"drone_velocity_linear_" + d: d for d in coords})
                vel.to_csv(os.path.join(out_path_vel, filename))

                acc: pd.DataFrame = df[["elapsed_time"] + ["accel_" + d for d in coords]]
                acc.rename({"accel_" + d: d for d in coords})
                acc.to_csv(os.path.join(out_path_acc, filename))


def scale_by(df: pd.DataFrame, coords: list[str], max):
    df[coords] = df[coords] / max

def max_mag(df: pd.DataFrame, coords: list[str]):
    coord_data = df[coords]

    sum_of_squares = (coord_data**2).sum(axis=1)
    magnitudes = np.sqrt(sum_of_squares)

    return magnitudes.max()


def walk_and_normalize(root: str, out: str, coords: list[str]):
    os.makedirs(out, exist_ok=True)

    max = 0
    for dirname in os.listdir(root):
        for filename in os.listdir(os.path.join(root, dirname)):
            curr = max_mag(
                pd.read_csv(
                    os.path.join(root, dirname, filename), 
                    usecols=["timestamp"] + coords
                    ), 
                coords
            )

            max = curr if curr > max else max
    
    for dirname in os.listdir(root):
        os.makedirs(os.path.join(out, dirname))
        for filename in os.listdir(os.path.join(root, dirname)):
            df = pd.read_csv(
                os.path.join(root, dirname, filename), 
                usecols=["timestamp"] + coords
            )
            scale_by(df, coords, max)
            df.to_csv(os.path.join(out, dirname, filename))


def main():
    data_root = "data/raw/drone-racing-dataset"
    pos_path = "data/clean/drone-racing-dataset/pos"
    vel_path = "data/clean/drone-racing-dataset/vel" 
    acc_path = "data/clean/drone-racing-dataset/acc"

    walk_and_process(
        root=data_root,
        out_path_pos=pos_path,
        out_path_vel=vel_path,
        out_path_acc=acc_path
    )
    print("Done extracting data")

    # pos_norm_path = "data/position/max_norm"
    # walk_and_normalize(
    #     root=pos_path,
    #     out=pos_norm_path,
    #     coords=["tx", "ty", "tz"]
    # )
    # print("Done normalizing position.")

    # vel_norm_path = "data/velocity/max_norm"
    # walk_and_normalize(
    #     root=vel_path,
    #     out=vel_norm_path,
    #     coords=["vx", "vy", "vz"]
    # )
    # print("Done normalizing velocity.")

    # acc_norm_path = "data/acceleration/max_norm"
    # walk_and_normalize(
    #     root=acc_path,
    #     out=acc_norm_path,
    #     coords=["ax", "ay", "az"]
    # )
    # prInt("Done normalizing acceleration.")
    
    print("Finished.")
    
if __name__ == "__main__":
    main()