from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd


@dataclass
class MMMData:
    df: pd.DataFrame
    channels: List[str]
    y: np.ndarray
    X: np.ndarray
    t: np.ndarray
    y_mu: float
    y_sd: float
    y_z: np.ndarray
    X_z: np.ndarray
    sin1: np.ndarray
    cos1: np.ndarray
    sin2: np.ndarray
    cos2: np.ndarray

    @classmethod
    def from_csv(cls, path: str, channels: List[str]) -> "MMMData":
        df = pd.read_csv(path)
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.sort_values("Date").reset_index(drop=True)

        y = df["Sales"].to_numpy(dtype=np.float64)
        X = df[channels].to_numpy(dtype=np.float64)
        t = np.arange(len(df), dtype=np.float64)

        # Standardize
        y_mu, y_sd = y.mean(), y.std() + 1e-6
        y_z = (y - y_mu) / y_sd

        X_mu = X.mean(axis=0)
        X_sd = X.std(axis=0) + 1e-6
        X_z = (X - X_mu) / X_sd

        # Seasonality (weekly data -> 52)
        w = 2 * np.pi * t / 52
        sin1, cos1 = np.sin(w), np.cos(w)
        sin2, cos2 = np.sin(2 * w), np.cos(2 * w)

        return cls(df, channels, y, X, t, y_mu, y_sd, y_z, X_z, sin1, cos1, sin2, cos2)

    def split(self, test_frac: float = 0.2) -> Tuple[np.ndarray, np.ndarray]:
        n = len(self.df)
        split = int(n * (1 - test_frac))
        return np.arange(split), np.arange(split, n)

    def take(self, idx: np.ndarray) -> dict:
        return {
            "y_z": self.y_z[idx],
            "X_z": self.X_z[idx],
            "t": self.t[idx],
            "sin1": self.sin1[idx],
            "cos1": self.cos1[idx],
            "sin2": self.sin2[idx],
            "cos2": self.cos2[idx],
        }
