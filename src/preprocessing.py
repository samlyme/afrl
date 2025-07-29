
import os
import numpy as np
import pandas as pd


def walk_and_process(
    root: str, 
    out_path_pos: str, 
    out_path_vel: str, 
    out_path_acc: str
):
    os.makedirs(out_path_pos, exist_ok=True)
    os.makedirs(out_path_vel, exist_ok=True)
    os.makedirs(out_path_acc, exist_ok=True)
    for dirpath, _, filenames in os.walk(root):
        # Position is always resampled
        for filename in filenames:
            if "500hz_freq_sync.csv" in filename:
                df: pd.DataFrame = pd.read_csv(os.path.join(dirpath, filename))

                # From the racing dataset
                coords = ["x", "y", "z"]
                tasks = [
                    ("drone_",                   out_path_pos),
                    ("drone_velocity_linear_",   out_path_vel),
                    ("accel_",                   out_path_acc),
                ]

                for prefix, out_path in tasks:
                    # build the column‐rename dict {"prefixx": "x", ...}
                    mapping = {f"{prefix}{d}": d for d in coords}

                    # select timestamp + all prefixed coords
                    cols = ["timestamp", *mapping.keys()]

                    # slice, rename, and dump
                    df_sub = df[cols].rename(columns=mapping)
                    df_sub.to_csv(
                        os.path.join(out_path, filename),
                        index=False
                    )


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
    
    print("Finished.")
    
if __name__ == "__main__":
    main()