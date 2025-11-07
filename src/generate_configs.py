# creates a grid of hyper params

from src.models import Config


hidden_state_dim_arr = [32, 64, 128, 256, 512, 1024]
num_gru_layers_arr = [1, 2, 3, 4, 5]
prediction_sequence_length_arr = [10, 20, 30, 40,50]

def save_config(config: Config, output_file: str | None = None):
    if not output_file:
        output_file = f"model_configs/gru_{config.hidden_state_dim}_{config.num_gru_layers}_{config.prediction_sequence_length}.json"

    with open(output_file, "w", encoding="utf-8") as f:
        f.write(config.model_dump_json(indent=2))

for h in hidden_state_dim_arr:
    for n in num_gru_layers_arr:
        for p in prediction_sequence_length_arr:
            config = Config(
                hidden_state_dim=h,
                num_gru_layers=n,
                prediction_sequence_length=p
            )
            save_config(config)

