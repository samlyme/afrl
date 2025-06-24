import h5py
from h5py import Group, Dataset

f = h5py.File("data/mid-air/MidAir/Kite_training/cloudy/sensor_records.hdf5", "r")

keys = list(f.keys())

# Jank but we know these are groups
grp: Group = f[keys[0]] # type: ignore

print(grp.keys())

ground: Group = grp["groundtruth"] # type: ignore
print(ground.keys())

pos: Dataset = ground["position"] # type: ignore

print(pos.shape)
print("pos at zero: ", pos[0])

gts: list[str] = []
def get_positions(name: str):
    if "groundtruth/position" in name:
        gts.append(name)
        
f.visit(get_positions)
print("visit gts", gts, type(gts))