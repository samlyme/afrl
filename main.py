#!/usr/bin/env python
# coding: utf-8

# # Main
# 
# This file contains everything you need to run the model. This requires already be in csv format in the "data/clean" directory.
# 
# You must run everything in "scripts" before running this file.

# In[1]:


import os
import numpy as np
import pandas as pd
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm.notebook import tqdm


# ## Split data
# 
# 5-fold stratified cross validation. Must be stratified because training data comes from different distributions.

# In[6]:


from math import floor
import random
from typing import TypedDict


class Fold(TypedDict):
    train: list[str]
    validation: list[str]
    test: list[str]

root = "data/velocity/max_norm"
strata = os.listdir(root)
k: int = 5
shuffle = False

folds: list[Fold] = [
    {"train": [], "validation": [], "test": []}
    for _ in range(k)
]
# Assume all csv's have unique names
for stratum in strata:
    files = os.listdir(os.path.join(root, stratum))
    if shuffle:
        random.shuffle(files)

    m = len(files)

    for i in range(k):
        fold_start = floor(m * (i/k))
        fold_end = floor(m * ((i+1)/k))

        fold = files[fold_start:fold_end]

        fold_train = floor(len(fold) * 0.65)
        fold_validation = floor(len(fold) * 0.85)

        folds[i]["train"].extend([os.path.join(root, stratum, f) for f in fold[0 : fold_train]])
        folds[i]["validation"].extend([os.path.join(root, stratum, f) for f in fold[fold_train : fold_validation]])
        folds[i]["test"].extend([os.path.join(root, stratum, f) for  f in fold[fold_validation :]])


# ### This class defines our custom trajectory dataset
# 
# I optimized it in the most practical way, balancing ram usage and conversion overheads.

# In[5]:


class TrajectoryDataset(torch.utils.data.Dataset):

    def __init__(self, files: list[str], input_length: int, output_length: int):
        self.input_length = input_length
        self.output_length = output_length
        self.total_sequence_length = input_length + output_length

        self.sample_map: list[tuple[int, int, int]] = []

        self.data_arrays: list[np.ndarray] = [] 

        current_global_index = 0

        for df_idx, file in enumerate(files):
            try:
                df: pd.DataFrame = pd.read_csv(file, usecols=["vx", "vy", "vz"])
            except Exception as e:
                print(f"Error reading {file}: {e}. Skipping.")
                continue

            numeric_cols = df.select_dtypes(include=['number']).columns
            if numeric_cols.empty:
                print(f"{file} has no numeric columns. Skipping.")
                continue

            data_array = df[numeric_cols].values.astype(np.float32) # Ensure float32 here

            if len(data_array) < self.total_sequence_length:
                print(f"{file} is too short ({len(data_array)} rows) for input_length={input_length} and output_length={output_length}. Skipping.")
                continue

            num_sequences_in_df = len(data_array) - self.total_sequence_length + 1

            for i in range(num_sequences_in_df):
                self.sample_map.append((current_global_index + i, df_idx, i))

            current_global_index += num_sequences_in_df
            self.data_arrays.append(data_array) # Store the NumPy array

        self.total_samples = current_global_index

    def __len__(self) -> int:
        return self.total_samples

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        if not (0 <= index < self.total_samples):
            raise IndexError(f"Index {index} is out of bounds for dataset of size {self.total_samples}")

        global_index, df_idx, local_start_row = self.sample_map[index]

        data_array = self.data_arrays[df_idx] # Retrieve NumPy array

        x_start_local = local_start_row
        x_end_local = local_start_row + self.input_length

        y_start_local = x_end_local
        y_end_local = y_start_local + self.output_length

        # Slice NumPy arrays (very fast)
        x_data = data_array[x_start_local:x_end_local]
        y_data = data_array[y_start_local:y_end_local]

        # Convert slices to PyTorch tensors (still happens in __getitem__, but from NumPy)
        # This conversion is very efficient from NumPy arrays
        x_tensor = torch.from_numpy(x_data) 
        y_tensor = torch.from_numpy(y_data)

        return x_tensor, y_tensor


# ### Our Model
# 
# Currently a basic GRU encoder-decoder.

# In[4]:


import torch
import torch.nn as nn

class TrajectoryPredictor(nn.Module):
    """
    An Encoder-Decoder model for trajectory prediction using GRU units.
    It takes an input sequence of points and predicts a future sequence of points.
    """
    def __init__(self, 
                 input_features_dim: int, 
                 hidden_state_dim: int, 
                 output_features_dim: int, 
                 num_gru_layers: int,
                 prediction_sequence_length: int):
        """
        Initializes the TrajectoryPredictor model.

        Args:
            input_features_dim (int): The number of features in each input time step
                                      (e.g., 2 for (x,y) coordinates).
            hidden_state_dim (int): The number of features in the hidden state of the GRU layers.
                                    This also determines the dimensionality of the context vector.
            output_features_dim (int): The number of features to predict at each output time step.
                                       (e.g., 2 for (x,y) coordinates).
            num_gru_layers (int): The number of stacked GRU layers for both encoder and decoder.
            prediction_sequence_length (int): The fixed number of future time steps to predict.
        """
        super().__init__() # Cleaner way to call super() in Python 3+

        self.hidden_state_dim = hidden_state_dim
        self.num_gru_layers = num_gru_layers
        self.prediction_sequence_length = prediction_sequence_length
        self.output_features_dim = output_features_dim

        self.encoder_gru = nn.GRU(input_features_dim, hidden_state_dim, num_gru_layers, batch_first=True)

        self.decoder_gru = nn.GRU(hidden_state_dim, hidden_state_dim, num_gru_layers, batch_first=True)

        self.output_projection_layer = nn.Linear(hidden_state_dim, output_features_dim)

    def forward(self, input_sequence: torch.Tensor) -> torch.Tensor:
        """
        Performs the forward pass of the TrajectoryPredictor.

        Args:
            input_sequence (torch.Tensor): The input trajectory sequence.
                                           Expected shape: (batch_size, input_seq_len, input_features_dim)

        Returns:
            torch.Tensor: The predicted future trajectory sequence.
                          Expected shape: (batch_size, prediction_sequence_length, output_features_dim)
        """
        device = input_sequence.device

        encoder_outputs, encoder_final_hidden_state = self.encoder_gru(input_sequence) 

        decoder_input_sequence = torch.zeros(
            input_sequence.size(0), 
            self.prediction_sequence_length, 
            self.hidden_state_dim 
        ).to(device) 

        decoder_outputs, _ = self.decoder_gru(decoder_input_sequence, encoder_final_hidden_state) 
        predicted_trajectory = self.output_projection_layer(decoder_outputs) 

        return predicted_trajectory


# ### Training
# 
# Setting up and running the training process.

# In[ ]:


from datetime import datetime


fold = folds[0]

X_len, y_len = 20, 10
train_dataset = TrajectoryDataset(fold['train'], X_len, y_len)
validation_dataset = TrajectoryDataset(fold['validation'], X_len, y_len)
test_dataset = TrajectoryDataset(fold['test'], X_len, y_len)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=10, shuffle=True)
validation_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=10, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=10, shuffle=True)


# In[ ]:


model = TrajectoryPredictor(
    input_features_dim=3,
    hidden_state_dim=64,
    output_features_dim=3,
    num_gru_layers=2,
    prediction_sequence_length=y_len
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("using device: ", device)
model.to(device)

# NOTE: When continuing training, remember to load the most recent model
model.load_state_dict(torch.load("best_models/model_20250709_185323_12"))


loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

def train_one_epoch(epoch_index, tb_writer):
    running_loss = 0.
    last_loss = 0.

    # Here, we use enumerate(training_loader) instead of
    # iter(training_loader) so that we can track the batch
    # index and do some intra-epoch reporting
    for i, data in tqdm(enumerate(train_loader), "train_dataset"):
        # Every data instance is an input + label pair
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)

        # Zero your gradients for every batch!
        optimizer.zero_grad()

        # Make predictions for this batch
        outputs = model(inputs)

        # Compute the loss and its gradients
        loss = loss_fn(outputs, labels)
        loss.backward()

        # Adjust learning weights
        optimizer.step()

        # Gather data and report
        running_loss += loss.item()
        if i % 1000 == 999:
            last_loss = running_loss / 1000 # loss per batch
            print('  batch {} loss: {}'.format(i + 1, last_loss))
            tb_x = epoch_index * len(train_loader) + i + 1
            tb_writer.add_scalar("Loss/train", last_loss, tb_x)
            running_loss = 0.

    return last_loss

# Initializing in a separate cell so we can easily add more epochs to the same run
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
writer = SummaryWriter("runs/afrl_trainer_{}".format(timestamp))

EPOCHS = 1000

best_vloss = 1_000_000.

for epoch in tqdm(range(EPOCHS), "epoch"):
    # Make sure gradient tracking is on, and do a pass over the data
    model.train(True)
    avg_loss = train_one_epoch(epoch, writer)


    running_vloss = 0.0
    # Set the model to evaluation mode, disabling dropout and using population
    # statistics for batch normalization.
    model.eval()

    # Disable gradient computation and reduce memory consumption.
    with torch.no_grad():
        for i, vdata in enumerate(validation_loader):
            vinputs, vlabels = vdata
            vinputs = vinputs.to(device)
            vlabels = vlabels.to(device)
            voutputs = model(vinputs)
            vloss = loss_fn(voutputs, vlabels)
            running_vloss += vloss

    avg_vloss = running_vloss / (i + 1)
    print(timestamp, 'LOSS train {} valid {}'.format(avg_loss, avg_vloss))

    # Log the running loss averaged per batch
    # for both training and validation
    writer.add_scalars('Training vs. Validation Loss',
                    { 'Training' : avg_loss, 'Validation' : avg_vloss },
                    epoch + 1)
    writer.flush()

    # Track best performance, and save the model's state
    if avg_vloss < best_vloss:
        best_vloss = avg_vloss
        model_path = os.path.join("best_models", 'model_{}_{}'.format(timestamp, epoch + 1))
        torch.save(model.state_dict(), model_path)


# ## Visualize model output
