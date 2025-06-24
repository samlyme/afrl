import h5py
from h5py import Group, Dataset
import pandas as pd

# MID-AIR position data is sampled at 100hz
path = "data/mid-air/MidAir/Kite_training/cloudy/sensor_records.hdf5"
f = h5py.File(path, "r")

gts: list[str] = []
def get_positions(name: str):
    if ("groundtruth/position" in name):
        gts.append(name)
        
f.visit(get_positions)

for gt in gts:
    print(gt)

item  = f[gts[0]]
if not isinstance(item, Dataset):
    raise TypeError("Not a dataset")

print("item", item)

df: pd.DataFrame = pd.DataFrame(item)
