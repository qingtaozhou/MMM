from typing import Optional, Tuple
from .utils import geometric_adstock_np, hill_np


import numpy as np
import pandas as pd
import pymc as pm
import pytensor
import pytensor.tensor as pt
from pytensor.scan import scan
import arviz as az

from mmm.data import MMMData
from mmm.utils import geometric_adstock_np, hill_np



# Force float64 everywhere (prevents scan dtype mismatches)
pytensor.config.floatX = "float64"


class BayesianMMM:
    def __init__(self, data: MMMData, train_idx: Optional[np.ndarray] = None):
        self.data = data
        self.train_idx = train_idx

        self.model: Optional[pm.Model] = None
        self.idata = None

        self.contrib_z: Optional[np.ndarray] = None
        self.mu_all: Optional[np.ndarray] = None

    # ----------------------------
    # Build model (TRAIN ONLY)
    # ----------------------------
    def build_model(self) -> pm.Model:
        d = self.data
        C = len(d.channels)

        if self.train_idx is None:
            ds = d.take(np.arange(len(d.df)))
        else:
            ds = d.take(self.train_idx)

        with pm.Model() as model:
            # Baseline
            intercept = pm.Normal("intercept", 0, 1)
            beta_trend = pm.Normal("beta_trend", 0, 0.5)

            beta_sin1 = pm.Normal("beta_sin1", 0, 0.5)
            beta_cos1 = pm.Normal("beta_cos1", 0, 0.5)
            beta_sin2 = pm.Normal("beta_sin2", 0, 0.3)
            beta_cos2 = pm.Normal("beta_cos2", 0, 0.3)

            trend_z = (ds["t"] - ds["t"].mean()) / (ds["t"].std() + 1e-6)
            baseline = (
                intercept
                + beta_trend * trend_z
                + beta_sin1 * ds["sin1"] + beta_cos1 * ds["cos1"]
                + beta_sin2 * ds["sin2"] + beta_cos2 * ds["cos2"]
            )

            # Media
            alpha = pm.Beta("alpha", 2, 2, shape=C) * 0.9
            half_sat = pm.LogNormal("half_sat", np.log(1.0), 0.5, shape=C)
            slope = pm.LogNormal("slope", np.log(1.2), 0.4, shape=C)
            beta_media = pm.HalfNormal("beta_media", 1.0, shape=C)

            Xz = pt.as_tensor_variable(ds["X_z"]).astype("float64")

            def step(x_t, prev, a):
                # keep float64 flow
                x_t = pt.cast(x_t, "float64")
                prev = pt.cast(prev, "float64")
                a = pt.cast(a, "float64")
                return x_t + a * prev

            media_terms = []
            for c in range(C):
                adstocked, _ = scan(
                    fn=step,
                    sequences=Xz[:, c],
                    outputs_info=pt.zeros((), dtype="float64"),
                    non_sequences=pt.cast(alpha[c], "float64"),
                )
                x_pos = pt.maximum(adstocked + 2.0, 0.0)
                sat = (x_pos ** slope[c]) / (x_pos ** slope[c] + half_sat[c] ** slope[c] + 1e-12)
                media_terms.append(beta_media[c] * sat)

            media = pt.sum(pt.stack(media_terms, axis=1), axis=1)
            mu = baseline + media

            sigma = pm.HalfNormal("sigma", 1.0)
            pm.Normal("y_obs", mu, sigma, observed=ds["y_z"])

        self.model = model
        return model

    # ----------------------------
    # Fit
    # ----------------------------
    def fit(
        self,
        draws: int = 1500,
        tune: int = 1500,
        chains: int = 4,
        target_accept: float = 0.9,
        random_seed: int = 7,
    ):
        if self.model is None:
            self.build_model()

        with self.model:
            self.idata = pm.sample(
                draws=draws,
                tune=tune,
                chains=chains,
                target_accept=target_accept,
                random_seed=random_seed,
            )
        return self.idata

    # ----------------------------
    # Contributions + ROAS (FULL DATA)
    # ----------------------------
    def compute_contributions_and_roas(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        if self.idata is None:
            raise RuntimeError("Call fit() before compute_contributions_and_roas().")

        d = self.data
        post = self.idata.posterior

        alpha = post["alpha"].mean(("chain", "draw")).values
        half = post["half_sat"].mean(("chain", "draw")).values
        slope = post["slope"].mean(("chain", "draw")).values
        beta = post["beta_media"].mean(("chain", "draw")).values

        n, C = d.X_z.shape
        contrib_z = np.zeros((n, C), dtype=np.float64)

        for c in range(C):
            ad = geometric_adstock_np(d.X_z[:, c], float(alpha[c]))
            sat = hill_np(ad + 2.0, float(half[c]), float(slope[c]))
            contrib_z[:, c] = float(beta[c]) * sat

        self.contrib_z = contrib_z
        contrib_sales = contrib_z * d.y_sd

        out = pd.concat(
            [
                d.df[["Date"] + d.channels + ["Sales"]],
                pd.DataFrame(contrib_sales, columns=[f"{c}_incremental_sales" for c in d.channels]),
            ],
            axis=1,
        )

        total_spend = d.X.sum(axis=0).astype(float)
        total_incr = contrib_sales.sum(axis=0).astype(float)
        roas = total_incr / (total_spend + 1e-12)

        summary = (
            pd.DataFrame(
                {
                    "Channel": d.channels,
                    "Total Spend": total_spend,
                    "Total Incremental Sales": total_incr,
                    "ROAS": roas,
                }
            )
            .sort_values("ROAS", ascending=False)
            .reset_index(drop=True)
        )

        return out, summary

    # ----------------------------
    # Holdout validation
    # ----------------------------
    def holdout_eval(self, test_frac: float = 0.2) -> Tuple[float, float]:
        if self.idata is None:
            raise RuntimeError("Call fit() before holdout_eval().")
        if self.contrib_z is None:
            raise RuntimeError("Call compute_contributions_and_roas() before holdout_eval().")

        d = self.data
        _, test_idx = d.split(test_frac)

        post = self.idata.posterior
        intercept = float(post["intercept"].mean())
        beta_trend = float(post["beta_trend"].mean())
        beta_sin1 = float(post["beta_sin1"].mean())
        beta_cos1 = float(post["beta_cos1"].mean())
        beta_sin2 = float(post["beta_sin2"].mean())
        beta_cos2 = float(post["beta_cos2"].mean())

        trend_z = (d.t - d.t.mean()) / (d.t.std() + 1e-6)
        baseline = (
            intercept
            + beta_trend * trend_z
            + beta_sin1 * d.sin1 + beta_cos1 * d.cos1
            + beta_sin2 * d.sin2 + beta_cos2 * d.cos2
        )

        media = self.contrib_z.sum(axis=1)
        mu = baseline + media
        self.mu_all = mu

        rmse = float(np.sqrt(np.mean((d.y_z[test_idx] - mu[test_idx]) ** 2)))
        mae = float(np.mean(np.abs(d.y_z[test_idx] - mu[test_idx])))
        return rmse, mae

    # ----------------------------
    # Save and Load
    # ----------------------------
    def save(self, filepath: str):
        """Save the fitted model's inference data."""
        if self.idata is None:
            raise RuntimeError("No fitted model to save. Call fit() first.")
        self.idata.to_netcdf(filepath)

    def load(self, filepath: str):
        """Load a previously saved model."""
        self.idata = az.from_netcdf(filepath)
