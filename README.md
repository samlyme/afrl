# AFRL Reseach

Research for drone trajectory prediction.

## TODO:
- Implement a "true" test of model accuracy that integrates the velocity prediction to find position.
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

TLDR: Run the get data script to get data.

```python
python src/get_data.py
```

Inside the `data/scripts` directory, you will find 4 scripts, each corresponding to a different dataset.

## Preprocessing

The data preprocessing is run in the notebook at the root of the project directory. 

## Create folds

When creating folds, we must be careful to not incure data leakage between folds.
Thus, we split the data into folds by files. 
```
python src/datasets.py data/velocity/max_norm --shuffle --seed 42
```
Replace `data/velocity/max_norm` with `data/position/max_norm` for position data.