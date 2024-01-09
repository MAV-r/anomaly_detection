import pandas as pd
from dvc.api import DVCFileSystem
from omegaconf import DictConfig
from sklearn.model_selection import train_test_split


def datasets_preparation(cfg: DictConfig) -> None:
    DVCFileSystem().get("data", "data", recursive=True)

    if cfg.dataset.process == "long":

        df = pd.read_parquet("./data/merged_dataset.parquet")

        X_ohe = pd.get_dummies(df.drop(cfg.dataset.target_name, axis=1), drop_first=True)

        X_train_pos, X_test, y_train_pos, y_test = train_test_split(
            X_ohe,
            df[cfg.dataset.target_name],
            test_size=cfg.dataset.test_size,
            random_state=cfg.dataset.random_state,
        )

        X_train, X_val, y_train, y_val = train_test_split(
            X_train_pos,
            y_train_pos,
            test_size=cfg.dataset.val_size,
            random_state=cfg.dataset.random_state,
        )

        X_test.to_parquet(".data/X_test.parquet")
        y_test.to_parquet(".data/y_test.parquet")
        X_val.to_parquet(".data/X_val.parquet")
        y_val.to_parquet(".data/y_val.parquet")
        X_train.to_parquet(".data/X_train.parquet")
        y_train.to_parquet(".data/y_train.parquet")

    elif cfg.dataset.process == "short":
        pass
        # print("short")
