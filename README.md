# AFRL Reseach

Research for drone trajectory prediction.

## TODO:
- Move preprocessing scripts outside of notebooks.
- Implement a "true" test of model accuracy that integrates the velocity prediction to find position.
- Refactor scripts into one python module.
- Implement model hyperparameter configuration.

## Dependencies

The dependencies of this project are managed with `uv` and python virtual environments. 

Install `uv` (if not already installed):
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Install dependencies and activate virtual environment.
```bash
uv sync
source .venv/bin/activate
```

## Getting the data

All the data is stored in the `data` directory. 

Think of this as a "staging area" where we can convert all the data into a consistent format.

It also by default stores all intermediate steps.

TLDR: Run the `fpv-uzh.py`, `mid-air.py`, and `riotu-labs.py` scripts, and useable `.csv` files will be produced in the `data/clean` directory. Make sure you run these scripts from the root of the project. The `generate_random_trajectories.py` script is optional and requires extra configuration.

```python
python data/scripts/fpv-uzh.py
python data/scripts/mid-air.py
python data/scripts/riotu-labs.py
```

Inside the `data/scripts` directory, you will find 4 scripts, each corresponding to a different dataset.

### FPV UZH

The `fpv-uzh.py` script is responsible for downloading, extracting, and cleaning the FPV UZH dataset.

This script first downloads the dataset which is initially received as `.zip` files and stored in the `data/dirty/fpv-uzh/archives` directory. It then expands the zip files, to `data/dirty/fpv-uzh/raw`. Then, it will read and reformat the `groundtruth.txt` files and stores it in `data/clean/fpv-uzh` as standard `.csv` files.

This is the desired positional data from the FPV UZH dataset.

### Mid-Air

The `mid-air.py` script is responsible for downloading, extracting, and cleaning the Mid-air dataset.

This script first downloads the dataset which is initially received as `.zip` files and stored in the `data/dirty/mid-air/MidAir` directory. It then expands the zip files. T Then, it will read and reformat the `groundtruth.txt` files and stores it in `data/clean/fpv-uzh` as standard `.csv` files.

Note: The mid-air data is stored as `.hdf5` files. Make sure you have the correct dependencies installed.

### Riotu Labs

The `riotu-labs.py` script downloads usable `.csv` files directly into `data/clean/riotu-labs`. The format is already correct and usable.

### Generated trajectories

The `generate_random_trajectories.py` script generates random trajectories and requires some extract parameters. Learn more by reading the code.

## Things done

- Data automation
- Encoder decoder architecture
- Stratified k-fold cross validation
- Sophisticated data normalization (statistical whitening)
- Out of distribution testing

## Things to be explored

- Batch normalization
- Other "distribution agnostic" approaches