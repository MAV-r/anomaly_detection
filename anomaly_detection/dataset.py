import pandas as pd
from dvc.api import DVCFileSystem
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from omegaconf import DictConfig
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def datasets_preparation(cfg: DictConfig) -> None:
    DVCFileSystem().get("data", "data", recursive=True)

    if cfg.dataset.process == "long":
        rus = RandomUnderSampler(random_state=cfg.dataset.random_state)
        sm = SMOTE(
            sampling_strategy=0.5, random_state=cfg.dataset.random_state, k_neighbors=5
        )

        df = pd.read_parquet("./data/merged_dataset.parquet")

        X_train_pre, X_test, y_train_pre, y_test = train_test_split(
            df.drop(cfg.dataset.target_name, axis=1),
            df[cfg.dataset.target_name],
            test_size=cfg.dataset.test_size,
            random_state=cfg.dataset.random_state,
        )

        X_train_sampl, y_train_sampl = sm.fit_resample(X_train_pre, y_train_pre)

        X_train, y_train = rus.fit_resample(X_train_sampl, y_train_sampl)

        scaler = StandardScaler()
        scaler.fit(X_train)

        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)

        X_train, X_val, y_train, y_val = train_test_split(
            X_train,
            y_train,
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
