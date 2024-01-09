import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from anomaly_detection.dataset import datasets_preparation
from anomaly_detection.model import FraudNet, train


def main():

    datasets_preparation()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = FraudNet().to(device)

    X_train = pd.read_parquet("./data/X_train.parquet")
    y_train = pd.read_parquet("./data/y_train.parquet")
    X_val = pd.read_parquet("./data/X_val.parquet")
    y_val = pd.read_parquet("./data/y_val.parquet")

    train_dataset = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(np.array(y_train), dtype=torch.float32),
    )
    val_dataset = TensorDataset(
        torch.tensor(X_val, dtype=torch.float32),
        torch.tensor(np.array(y_val), dtype=torch.float32),
    )

    train_loader = DataLoader(
        train_dataset, batch_size=64, shuffle=True, num_workers=2, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=64, shuffle=False, num_workers=2, pin_memory=True
    )

    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)

    train(net, optimizer, 1, train_loader, val_loader)  # , scheduler=scheduler

    torch.save(net.state_dict(), "./models/")


if __name__ == "__main__":
    main()
