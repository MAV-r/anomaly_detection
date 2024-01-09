# import lightning.pytorch as pl
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch import nn


class FraudNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.save_hyperparameters()
        self.loss_fn = nn.BCELoss()

        self.model = nn.Sequential(
            nn.Linear(in_features=2818, out_features=1500),
            nn.ReLU(),
            nn.Linear(in_features=1500, out_features=2000),
            nn.ReLU(),
            nn.Dropout(p=0.25),
            nn.Linear(in_features=2000, out_features=1000),
            nn.ReLU(),
            nn.Linear(in_features=1000, out_features=800),
            nn.ReLU(),
            nn.Linear(in_features=800, out_features=1),
            nn.Sigmoid(),
        )

    def forward(self, data):
        return self.model(data)


def test(model, loader, last):
    loss_log = []
    roc_auc_log = []
    pred = []
    true = []
    model.eval()
    loss_func = nn.BCELoss()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    for data, target in loader:

        data = data.to(device)
        target = target.to(device)

        with torch.no_grad():
            logits = model(data)
            loss = loss_func(logits, target)

        y_true = target.cpu()
        roc_auc_log.append(roc_auc_score(y_true, logits.cpu().detach().numpy()))
        loss_log.append(loss.item())

        if last:
            y_pred = [1 if x >= 0.5 else 0 for x in logits.cpu().detach().numpy()]
            pred.extend(y_pred)
            true.extend([int(x) for x in y_true])

    return np.mean(loss_log), np.mean(roc_auc_log), true, pred


def train_epoch(model, optimizer, train_loader, last):
    loss_log = []
    roc_auc_log = []
    pred = []
    true = []
    model.train()
    loss_func = nn.BCELoss()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    for data, target in train_loader:
        data = data.to(device)
        target = target.to(device)

        optimizer.zero_grad()
        logits = model(data)

        loss = loss_func(logits, target)
        loss.backward()
        optimizer.step()

        y_true = target.cpu()
        roc_auc_log.append(roc_auc_score(y_true, logits.cpu().detach().numpy()))
        loss_log.append(loss.item())

        if last:
            y_pred = [1 if x >= 0.5 else 0 for x in logits.cpu().detach().numpy()]
            pred.extend(y_pred)
            true.extend([int(x) for x in y_true])

    return loss_log, roc_auc_log, true, pred


def train(model, optimizer, n_epochs, train_loader, val_loader, scheduler=None):
    train_loss_log, train_roc_auc_log, val_loss_log, val_roc_auc_log = [], [], [], []

    for epoch in range(n_epochs):
        if epoch == n_epochs - 1:
            train_loss, train_roc_auc, _, _ = train_epoch(
                model, optimizer, train_loader, last=True
            )
            val_loss, val_roc_auc, _, _ = test(model, val_loader, last=True)
        else:
            train_loss, train_roc_auc, _, _ = train_epoch(
                model, optimizer, train_loader, last=False
            )
            val_loss, val_roc_auc, _, _ = test(model, val_loader, last=False)

        train_loss_log.extend(train_loss)
        train_roc_auc_log.extend(train_roc_auc)

        val_loss_log.append(val_loss)
        val_roc_auc_log.append(val_roc_auc)

        if scheduler is not None:
            scheduler.step()

    return train_loss_log, train_roc_auc_log, val_loss_log, val_roc_auc_log
