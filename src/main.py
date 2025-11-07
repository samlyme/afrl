import argparse
from datetime import datetime
import os
import torch
import torch.utils.tensorboard

from src.datasets import Fold, Split, TrajectoryDataset, read_split
from src.models import Config, TrajectoryPredictor, model_from_config, read_config

class EarlyStopping:
    """
    Stops training when monitored metric hasn't improved for `patience` epochs.
    Optionally restores best model weights.

    Args:
        patience (int): epochs to wait after last improvement.
        min_delta (float): minimum change to qualify as improvement (absolute).
        mode (str): 'min' for loss, 'max' for accuracy/score.
        restore_best (bool): if True, restore model weights from best epoch.
    """
    def __init__(self, patience=10, min_delta=0.0, mode='min', restore_best=True):
        assert mode in ('min', 'max')
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.restore_best = restore_best

        self.best_score = None
        self.best_state = None
        self.num_bad_epochs = 0
        self.should_stop = False

    def _is_better(self, score, best):
        if best is None:
            return True
        if self.mode == 'min':
            return score < best - self.min_delta
        else:
            return score > best + self.min_delta

    def step(self, score, model=None):
        """
        Call this at the end of each epoch with the validation metric.
        Args:
            score (float): the monitored value (e.g., val_loss or val_acc).
            model (torch.nn.Module|Any): model to snapshot (optional).
        Returns:
            bool: True if training should stop.
        """
        if self._is_better(score, self.best_score):
            self.best_score = score
            self.num_bad_epochs = 0
            if model is not None and self.restore_best:
                # keep a lightweight copy of weights only
                import copy
                self.best_state = copy.deepcopy(getattr(model, "state_dict", lambda: {})())
        else:
            self.num_bad_epochs += 1
            if self.num_bad_epochs >= self.patience:
                self.should_stop = True
        return self.should_stop

    def restore(self, model):
        """Restore best weights if `restore_best=True` and a snapshot exists."""
        if self.restore_best and self.best_state is not None and model is not None:
            model.load_state_dict(self.best_state)



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
        epochs: int
    ):
        best_vloss = 1000000
        early_stopper = EarlyStopping(patience=10, min_delta=10e-6)
        for epoch in range(epochs):
            self.model.train(True)
            avg_loss = self.train_epoch(epoch) 

            running_vloss = 0.0
            # Set the model to evaluation mode, disabling dropout and using population
            # statistics for batch normalization.
            self.model.eval()
            # Disable gradient computation and reduce memory consumption.
            i = 0
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
            
            if early_stopper.step(avg_vloss, self.model):
                print(f"Early stopping")
                break

            # Track best performance, and save the model's state
            if avg_vloss < best_vloss:
                best_vloss = avg_vloss
                torch.save(self.model.state_dict(), os.path.join(self.model_path, f"model_{datetime.now()}"))

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
        "-c",
        "--config",
        type=str, 
        required=True,
        help="Specifies a model config file."
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

    config: Config = read_config(args.config)

    X_len, y_len = 20, config.prediction_sequence_length

    train_dataset = TrajectoryDataset(fold.train, X_len, y_len)
    validation_dataset = TrajectoryDataset(fold.validation, X_len, y_len)
    test_dataset = TrajectoryDataset(split.test, X_len, y_len)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=10, shuffle=True)
    validation_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=10, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=10, shuffle=True)


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model=model_from_config(config, 3)

    if args.model:
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

    trainer.train_epochs(1000)

if __name__ == "__main__":
    main()