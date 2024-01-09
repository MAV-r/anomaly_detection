import pandas as pd
from hydra import compose
from dvc.api import DVCFileSystem
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

import hydra
from omegaconf import DictConfig

@hydra.main(config_path="../configs", config_name="config", version_base="1.3")
def datasets_preparation(cfg: DictConfig) -> None:

    DVCFileSystem().get("data", "data", recursive=True)

    if cfg.dataset.process=="long":
        rus = RandomUnderSampler(random_state=42)
        sm = SMOTE(sampling_strategy=0.5, random_state=42, k_neighbors=5)

        df = pd.read_parquet('./data/merged_dataset.parquet')

        X_train_pre, X_test, y_train_pre, y_test = train_test_split(df.drop('isFraud', axis=1), df['isFraud'], test_size=0.3,
                                                            random_state=42)

        X_train_sampl, y_train_sampl = sm.fit_resample(X_train_pre, y_train_pre)

        X_train, y_train = rus.fit_resample(X_train_sampl, y_train_sampl)

        scaler = StandardScaler()
        scaler.fit(X_train)

        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)

        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42)

        X_test.to_parquet('.data/X_test.parquet')
        y_test.to_parquet('.data/y_test.parquet')
        X_val.to_parquet('.data/X_val.parquet')
        y_val.to_parquet('.data/y_val.parquet')
        X_train.to_parquet('.data/X_train.parquet')
        y_train.to_parquet('.data/y_train.parquet')

    elif cfg.dataset.process=="short":
        # pass
        print("short")


