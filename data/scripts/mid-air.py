import os
import subprocess
import h5py
from h5py import Dataset
import numpy as np
import pandas as pd


def download_and_unzip(out_dir = "data/dirty/mid-air", download_config = "data/scripts/mid-air_sources.txt"):
    # Ensure the output directory exists
    os.makedirs(out_dir, exist_ok=True)

    # Download files using wget
    wget_command = [
        "wget",
        "--content-disposition",
        "-x",
        "-nH",
        "-P",
        out_dir,
        "-i",
        download_config,
    ]
    subprocess.run(wget_command, check=True)

    # Find and unzip files
    for root, _, files in os.walk(out_dir):
        for filename in files:
            if filename.endswith(".zip"):
                zip_filepath = os.path.join(root, filename)
                unzip_command = ["unzip", "-o", "-d", os.path.dirname(zip_filepath), zip_filepath]
                subprocess.run(unzip_command, check=True)

# MID-AIR position data is sampled at 100hz
def process_hdf5(path: str) -> list[pd.DataFrame]:
    f = h5py.File(path, "r")

    gts: list[str] = []
    def get_positions(name: str):
        if ("groundtruth/position" in name):
            gts.append(name)
            
    f.visit(get_positions)

    item  = f[gts[0]] 
    if not isinstance(item, Dataset):
        raise TypeError("Not a dataset")

    out = []
    for gt in gts:
        curr = f[gt]
        if not isinstance(curr, Dataset): continue
        df = pd.DataFrame(curr, columns=["tx", "ty", "tz"])

        # This dataset is sampled at 100hz
        interval =  1/100
        rows: int = len(df)
        timestamps = np.arange(0, rows * interval, interval)
        df.insert(0, "timestamp", timestamps)
        out.append(df)

    return out

def walk_and_process(path: str) -> list[pd.DataFrame]:
    out = []
    for dirpath, dirnames, filenames in os.walk(path):
        for filename in filenames:
            if filename.endswith("hdf5"):
                out.extend(process_hdf5(os.path.join(dirpath, filename)))

    return out


def main():
    download_and_unzip()
    
    all_pos: list[pd.DataFrame] = walk_and_process("data/dirty/mid-air")
    out_dir = "data/clean/mid-air"
    os.makedirs(out_dir, exist_ok=True)

    for index, pos in enumerate(all_pos):
        pos.to_csv(os.path.join(out_dir, f"trajectory_{index}.csv"))

if __name__ == "__main__":
    main()