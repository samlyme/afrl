import argparse
from datetime import datetime
import os
import re
import torch
import torch.utils.tensorboard

from src.datasets import Fold, Split, TrajectoryDataset, read_split
from src.models import GRUTrajectoryPredictor


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
        device: torch.device,
        writer: torch.utils.tensorboard.SummaryWriter,
        model_path: str,
    ):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn

        self.train_loader = train_loader
        self.validation_loader = validation_loader
        self.test_loader = test_loader

        self.device = device
        print("using device ", self.device)

        self.writer = writer

        self.model_path = model_path
        os.makedirs(model_path, exist_ok=True)

    
    def train_epochs(
        self,
        epochs: int,
        start: int = 0
    ):
        best_vloss = 1000000
        for epoch in range(start, epochs):
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

            avg_vloss = running_vloss / (i + 1)  # type: ignore
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
                torch.save(self.model.state_dict(), os.path.join(self.model_path, f"{datetime.now()}_epoch_{epoch}.pt"))

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
        default=None,
        help="Specifies a modelfile to start from. If not specified, model will start from scratch."
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
    model = GRUTrajectoryPredictor(
        input_features_dim=3,
        hidden_state_dim=64,
        output_features_dim=3,
        num_gru_layers=2,
        prediction_sequence_length=y_len
    )

    start = 0
    if args.model:
        m = re.search(r"_epoch_(\d+)\.pt$", args.model)
        if not m:
            raise ValueError(f"Could not parse epoch from '{args.model}'")
        start = int(m.group(1))
        model.load_state_dict(torch.load(args.model))

    model.to(device)

    trainer = Trainer(
        model=model,
        optimizer=torch.optim.Adam(model.parameters(), lr=0.001),
        loss_fn=torch.nn.MSELoss(),
        train_loader=train_loader,
        validation_loader=validation_loader,
        test_loader=test_loader,
        device=device,
        writer=torch.utils.tensorboard.SummaryWriter(f"experiments/logs/{args.name}"),
        model_path=f"experiments/models/{args.name}"
    )

    trainer.train_epochs(1000, start)

if __name__ == "__main__":
    main()