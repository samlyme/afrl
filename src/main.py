import argparse
from datetime import datetime
import os
import torch
import torch.utils.tensorboard

from src.datasets import Fold, Split, TrajectoryDataset, read_split
from src.models import TrajectoryPredictor


class Trainer:
    """
    Instantiate an object of this class to train.
    """
    model: torch.nn.Module
    optimizer: torch.optim.Optimizer
    loss_fn: torch.nn.Module
    train_loader: torch.utils.data.DataLoader
    validation_loader: torch.utils.data.DataLoader
    test_loader: torch.utils.data.DataLoader
    device: torch.device
    writer: torch.utils.tensorboard.SummaryWriter

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        loss_fn: torch.nn.Module,
        train_loader: torch.utils.data.DataLoader,
        validation_loader: torch.utils.data.DataLoader,
        test_loader: torch.utils.data.DataLoader,
        device: torch.device | None = None,
        writer: torch.utils.tensorboard.SummaryWriter | None = None
    ):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn

        self.train_loader = train_loader
        self.validation_loader = validation_loader
        self.test_loader = test_loader

        if device:
            self.device = device
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("using device ", self.device)

        if writer:
            self.writer = writer
        else:
            self.writer = torch.utils.tensorboard.SummaryWriter(
                f"runs/afrl_trainer_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )

    
    def train_epochs(
        self,
        epochs: int
    ):
        best_vloss = 1000000
        for epoch in range(epochs):
            self.model.train(True)
            avg_loss = self.train_epoch(epoch) 

            running_vloss = 0.0
            # Set the model to evaluation mode, disabling dropout and using population
            # statistics for batch normalization.
            self.model.eval()
            # Disable gradient computation and reduce memory consumption.
            with torch.no_grad():
                for i, vdata in enumerate(self.validation_loader):
                    vinputs, vlabels = vdata
                    vinputs = vinputs.to(self.device)
                    vlabels = vlabels.to(self.device)
                    voutputs = self.model(vinputs)
                    vloss = self.loss_fn(voutputs, vlabels)
                    running_vloss += vloss

            avg_vloss = running_vloss / (i + 1)
            print(f"LOSS train {avg_loss} valid {avg_vloss}")
            
            # Log the running loss averaged per batch
            # for both training and validation
            self.writer.add_scalars('Training vs. Validation Loss',
                            { 'Training' : avg_loss, 'Validation' : avg_vloss },
                            epoch + 1)
            self.writer.flush()
            # TODO: implement early stopping

            # Track best performance, and save the model's state
            if avg_vloss < best_vloss:
                best_vloss = avg_vloss
                model_path = os.path.join("best_models", f"model_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
                torch.save(self.model.state_dict(), model_path)

    def train_epoch(
        self,
        epoch_index: int, 
    ):
        running_loss = 0.
        last_loss = 0.

        # Here, we use enumerate(training_loader) instead of
        # iter(training_loader) so that we can track the batch
        # index and do some intra-epoch reporting
        for i, data in enumerate(self.train_loader):
            # Every data instance is an input + label pair
            inputs, labels = data
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            # Zero your gradients for every batch!
            self.optimizer.zero_grad()

            # Make predictions for this batch
            outputs = self.model(inputs)

            # Compute the loss and its gradients
            loss = self.loss_fn(outputs, labels)
            loss.backward()

            # Adjust learning weights
            self.optimizer.step()

            # Gather data and report
            running_loss += loss.item()
            if i % 1000 == 999:
                last_loss = running_loss / 1000 # loss per batch
                print(f'  batch {i+1} loss: {last_loss}')
                tb_x = epoch_index * len(self.train_loader) + i + 1
                self.writer.add_scalar("Training Loss", last_loss, tb_x)
                running_loss = 0.

        return last_loss

# TODO: implement cli args
def main():
    parser = argparse.ArgumentParser(
        description="Train models with specific data and hyperparameters."
    )

    parser.add_argument(
        "name",
        type=str,
        help="The name of this training job. Used for tensorboard reporting."
    )

    parser.add_argument(
        "-s",
        "--split",
        type=str,
        required=True,
        help="Path to a valid .json file that specifies the data split for k-fold CV."
    )

    parser.add_argument(
        "-f",
        "--fold",
        type=int,
        required=True,
        help="Specifies the specific fold to train on. Must be within the range of folds in split. Zero-indexed."
    )

    parser.add_argument(
        "-m",
        "--model",
        type=str,
        help="Specifies a modelfile to start from. If not specified, model will start from scratch."
    )

    parser.add_argument(
        "-e",
        "--epoch",
        type=int,
        default=0,
        help="Specifies the starting epoch. Useful for when jobs are interupted."
    )

    
    args = parser.parse_args()
    

    split: Split = read_split(args.split)
    if args.fold >= len(split.folds):
        raise Exception("Invalid fold index.")

    fold: Fold = split.folds[args.fold]

    # TODO: (MAYBE) implement sequence lengths as hyperparameters
    X_len, y_len = 20, 10
    train_dataset = TrajectoryDataset(fold.train, X_len, y_len)
    validation_dataset = TrajectoryDataset(fold.validation, X_len, y_len)
    test_dataset = TrajectoryDataset(split.test, X_len, y_len)

    # TODO: (MAYBE) Implement custom sampler
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=10, shuffle=True)
    validation_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=10, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=10, shuffle=True)


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # TODO: Implement model config object that can be saved.
    model=TrajectoryPredictor(
        input_features_dim=3,
        hidden_state_dim=64,
        output_features_dim=3,
        num_gru_layers=2,
        prediction_sequence_length=y_len
    )
    model.load_state_dict(torch.load("best_models/model_20250711_172714"))
    model.to(device)

    trainer = Trainer(
        model=model,
        optimizer=torch.optim.Adam(model.parameters(), lr=0.001),
        loss_fn=torch.nn.MSELoss(),
        train_loader=train_loader,
        validation_loader=validation_loader,
        test_loader=test_loader,
        device=device
    )

    trainer.train_epochs(1000)

if __name__ == "__main__":
    main()