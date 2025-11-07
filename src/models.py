from pydantic import BaseModel, conint
import torch

class Config(BaseModel):
    hidden_state_dim: int 
    num_gru_layers: int
    prediction_sequence_length: int

def model_from_config(config: Config, feature_dim: int) -> torch.nn.Module:
    return TrajectoryPredictor(
        input_features_dim=feature_dim,
        hidden_state_dim=config.hidden_state_dim,
        output_features_dim=feature_dim,
        num_gru_layers=config.num_gru_layers,
        prediction_sequence_length=config.prediction_sequence_length
    )


def read_config(file_path: str) -> Config:
    with open(file_path, 'r', encoding='utf-8') as f:
        raw_data = f.read()
    
    try:
        return Config.model_validate_json(raw_data)

    except Exception as e:
        raise e

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