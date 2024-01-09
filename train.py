import numpy as np
import pandas as pd
import torch
from hydra import compose, initialize
from torch.utils.data import DataLoader, TensorDataset

from anomaly_detection.dataset import datasets_preparation
from anomaly_detection.model import FraudNet, train


def main():
    initialize(version_base=None, config_path="configs", job_name="anomaly_detection")
    cfg = compose(config_name="config")

    datasets_preparation(cfg)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = FraudNet().to(device)

    X_train = pd.read_parquet("./data/X_train.parquet")
    y_train = pd.read_parquet("./data/y_train.parquet")
    X_val = pd.read_parquet("./data/X_val.parquet")
    y_val = pd.read_parquet("./data/y_val.parquet")

    train_dataset = TensorDataset(
        torch.tensor(X_train.values, dtype=torch.float32),
        torch.tensor(np.array(y_train), dtype=torch.float32),
    )
    val_dataset = TensorDataset(
        torch.tensor(X_val.values, dtype=torch.float32),
        torch.tensor(np.array(y_val), dtype=torch.float32),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.trainer.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.trainer.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.params.learning_rate)

    train(model, optimizer, cfg.trainer.epochs, train_loader, val_loader)

    torch.save(model.state_dict(), "./models/")


if __name__ == "__main__":
    main()
