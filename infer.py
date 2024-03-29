import mlflow
import pandas as pd
import torch
from dvc.api import DVCFileSystem
from hydra import compose, initialize

from anomaly_detection.model import FraudNet


def main():
    initialize(version_base=None, config_path="configs", job_name="anomaly_detection")
    cfg = compose(config_name="config")
    DVCFileSystem().get("model", "model", recursive=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = FraudNet().to(device)
    model.load_state_dict(torch.load("./models/model"))
    model.eval()

    X_test = pd.read_parquet("./data/X_test.parquet")
    mlflow.set_tracking_uri(uri=cfg.ml_ops.mlflow_server)

    with mlflow.start_run():
        params = {
            "Epoch": cfg.trainer.epochs,
        }
        mlflow.log_params(params)
        with torch.no_grad():
            logits = model(X_test)

            # y_test_true = [int(x) for x in y_test.cpu().detach().numpy()]
            y_pred = [1 if x >= 0.5 else 0 for x in logits.cpu().detach().numpy()]

        pd.DataFrame(y_pred).to_csv("./data/predictions.csv")


if __name__ == "__main__":
    main()
