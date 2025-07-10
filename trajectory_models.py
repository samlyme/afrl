import numpy as np
import pandas as pd
import torch

class TrajectoryDataset(torch.utils.data.Dataset):
    """
    Dataset that is built from a list of paths to .csv files. 
    Currently assumes that you have velocity data.
    """
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

            data_array = df.values.astype(np.float32) # Ensure float32 here
            
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


class TrajectoryPredictor(torch.nn.Module):
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

        self.encoder = torch.nn.GRU(input_features_dim, hidden_state_dim, num_gru_layers, batch_first=True)
        
        self.decoder = torch.nn.GRU(hidden_state_dim, hidden_state_dim, num_gru_layers, batch_first=True)
        
        self.projection = torch.nn.Linear(hidden_state_dim, output_features_dim)

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

        encoder_outputs, encoder_final_hidden_state = self.encoder(input_sequence) 

        decoder_input_sequence = torch.zeros(
            input_sequence.size(0), 
            self.prediction_sequence_length, 
            self.hidden_state_dim 
        ).to(device) 

        decoder_outputs, _ = self.decoder(decoder_input_sequence, encoder_final_hidden_state) 
        predicted_trajectory = self.projection(decoder_outputs) 
        
        return predicted_trajectory