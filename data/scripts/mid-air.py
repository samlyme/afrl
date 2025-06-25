import os
import h5py
from h5py import Group, Dataset
import pandas as pd

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

    return [pd.DataFrame(f[gt], columns=["tx", "ty", "tz"]) for gt in gts if isinstance(f[gt], Dataset)] # type: ignore

def walk_and_process(path: str) -> list[pd.DataFrame]:
    out = []
    for dirpath, dirnames, filenames in os.walk(path):
        for filename in filenames:
            if filename.endswith("hdf5"):
                out.extend(process_hdf5(os.path.join(dirpath, filename)))

    return out


def main():
    all_pos: list[pd.DataFrame] = walk_and_process("data/mid-air")
    out_dir = "data/clean/mid-air"
    os.makedirs(out_dir, exist_ok=True)

    for index, pos in enumerate(all_pos):
        pos.to_csv(os.path.join(out_dir, f"trajectory_{index}.csv"))

if __name__ == "__main__":
    main()