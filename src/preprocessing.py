
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
    out_path_vel: str | None, 
    out_path_acc: str | None
):
    for dirname in os.listdir(root):
        # Position is always resampled
        os.makedirs(os.path.join(out_path_pos, dirname), exist_ok=True)
        if out_path_vel:
            os.makedirs(os.path.join(out_path_vel, dirname), exist_ok=True)
        if out_path_acc:
            os.makedirs(os.path.join(out_path_acc, dirname), exist_ok=True)

        for filename in os.listdir(os.path.join(root, dirname)):
            df = pd.read_csv(os.path.join(root, dirname, filename))

            pos = resample(df, 0.1)
            pos.to_csv(os.path.join(out_path_pos, dirname, filename))

            if out_path_vel:
                vel = pos_to_vel(pos)
                vel.to_csv(os.path.join(out_path_vel, dirname, filename))

            if out_path_acc:
                acc = vel_to_acc(vel)
                acc.to_csv(os.path.join(out_path_acc, dirname, filename))


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
    parser = argparse.ArgumentParser(
        description="Derive velocity and acceleration data. Normalize said data."
    )

    parser.add_argument(
        "--all",
        action="store_true",
        help="Derives vel and acc, then normalizes vel and acc."
        "NOTE: This does not specify which datasets to include, just what actions to perform on the datasets."
    )
    
    parser.add_argument(
        "-t",
        "--time",
        type=float,
        default=0.1,
        help="Resamples position data. Defaults to 0.1 sec or 10hz."
    )

    parser.add_argument(
        "-v",
        "--velocity",
        action="store_true",
        help="Derives velocity from position."
    )

    parser.add_argument(
        "-a",
        "--acceleration",
        action="store_true",
        help="Derives acceleration from velocity. Fails if velocity does not exist."
    )

    parser.add_argument(
        "-np",
        "--norm-position",
        action="store_true",
        help="Normalizes position data."
    )

    parser.add_argument(
        "-nv",
        "--norm-velocity",
        action="store_true",
        help="Normalizes velocity data."
    )

    parser.add_argument(
        "-na",
        "--norm-acceleration",
        action="store_true",
        help="Normalizes acceleration data."
    )

    args = parser.parse_args()

    data_root = "data/clean"
    pos_path = "data/position/raw"
    vel_path = "data/velocity/raw" 
    acc_path = "data/acceleration/raw"

    walk_and_process(
        root=data_root,
        out_path_pos=pos_path,
        out_path_vel=vel_path if args.all or args.velocity else None,
        out_path_acc=acc_path if args.all or args.acceleration else None
    )
    print("Done resampling and deriving.")

    if args.all or args.norm_position:
        pos_norm_path = "data/position/max_norm"
        walk_and_normalize(
            root=pos_path,
            out=pos_norm_path,
            coords=["tx", "ty", "tz"]
        )
        print("Done normalizing position.")

    if args.all or args.norm_velocity:
        vel_norm_path = "data/velocity/max_norm"
        walk_and_normalize(
            root=vel_path,
            out=vel_norm_path,
            coords=["vx", "vy", "vz"]
        )
        print("Done normalizing velocity.")

    if args.all or args.norm_acceleration:
        acc_norm_path = "data/acceleration/max_norm"
        walk_and_normalize(
            root=acc_path,
            out=acc_norm_path,
            coords=["ax", "ay", "az"]
        )
        print("Done normalizing acceleration.")
    
    print("Finished.")
    
if __name__ == "__main__":
    main()