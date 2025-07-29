import argparse
import os
import numpy as np
from sklearn.model_selection import KFold
import torch
import torch.utils.tensorboard

from src.datasets import generate_auto_encoder_dataset
from src.models import FlightAutoEncoder

class Trainer:
    """
    Instantiate an object of this class to train.
    """
    model: torch.nn.Module
    optimizer: torch.optim.Optimizer
    loss_fn: torch.nn.Module
    train_loader: torch.utils.data.DataLoader
    validation_loader: torch.utils.data.DataLoader
    # test_loader: torch.utils.data.DataLoader
    device: torch.device
    writer: torch.utils.tensorboard.SummaryWriter

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        loss_fn: torch.nn.Module,
        train_loader: torch.utils.data.DataLoader,
        validation_loader: torch.utils.data.DataLoader,
        # test_loader: torch.utils.data.DataLoader,
        device: torch.device,
        writer: torch.utils.tensorboard.SummaryWriter,
        model_path: str,
    ):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn

        self.train_loader = train_loader
        self.validation_loader = validation_loader
        # self.test_loader = test_loader

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
                torch.save(self.model.state_dict(), os.path.join(self.model_path, f"epoch_{epoch}.pt"))

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

def get_args():
    parser = argparse.ArgumentParser(
        description="Train models with specific data and hyperparameters."
    )

    parser.add_argument(
        "name",
        type=str,
        help="The name of this training job. Used for tensorboard reporting."
    )

    return parser.parse_args()

def main():
    vel_dir = "data/clean/drone-racing-dataset/vel"
    files = os.listdir(vel_dir)
    dataset: torch.utils.data.Dataset = generate_auto_encoder_dataset(
        files=[os.path.abspath(os.path.join(vel_dir, fname)) for fname in files],
        input_sr=500,
        target_sr=100,
        window_size=100,
        norm_out=False
    )
    print("dataset size: ", len(dataset))
    kf = KFold(n_splits=5, shuffle=True, random_state=1)
    folds = list(kf.split(np.arange(len(dataset))))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    args = get_args()

    def train_eval_fold(train_idx, val_idx, fold_idx):
        model =  FlightAutoEncoder(
            seq_len=100,
            in_dim=3,
            latent_dim=32,
        )
        model.to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        loss_fn = torch.nn.MSELoss()
        train_loader = torch.utils.data.DataLoader(
            torch.utils.data.Subset(dataset, train_idx),
            batch_size=32,
            shuffle=True
        )
        validation_loader = torch.utils.data.DataLoader(
            torch.utils.data.Subset(dataset, val_idx),
            batch_size=32,
        )

        logdir = f"experiments/logs/{args.name}/fold_{fold_idx}"
        model_path = f"experiments/models/{args.name}/fold_{fold_idx}"
        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            loss_fn=loss_fn,
            train_loader=train_loader,
            validation_loader=validation_loader,
            device=device,
            writer=torch.utils.tensorboard.SummaryWriter(logdir),
            model_path=model_path
        )

        trainer.train_epochs(50)

    for i, (train_idx, val_idx) in enumerate(folds):
        train_eval_fold(train_idx, val_idx, i)

if __name__ == "__main__":
    main()