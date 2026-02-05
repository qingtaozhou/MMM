import numpy as np
import pandas as pd

def geometric_adstock(x: np.ndarray, alpha: float) -> np.ndarray:
    """
    Simple geometric adstock:
      y[t] = x[t] + alpha * y[t-1]
    alpha in [0, 1). Higher => longer carryover.
    """
    y = np.zeros_like(x, dtype=float)
    for t in range(len(x)):
        y[t] = x[t] + (alpha * y[t-1] if t > 0 else 0.0)
    return y

def hill_saturation(x: np.ndarray, half_saturation: float, slope: float = 1.0) -> np.ndarray:
    """
    Hill-type saturation (diminishing returns):
      f(x) = x^s / (x^s + k^s)
    Returns values in [0, 1].
    """
    x = np.clip(x, 0, None)
    
    xs = np.power(x, slope)
    ks = np.power(half_saturation, slope)
    return xs / (xs + ks + 1e-12)

def simulate_mmm_like_data(
    start="2018-01-07",
    end="2021-10-31",
    seed=7
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    # Weekly dates data for the spend
    dates = pd.date_range(start=start, end=end, freq="W-SUN")
    n = len(dates)
    t = np.arange(n)

    # (1) Spend generation with many zeros
    # "Flighting" (campaign on/off) using Bernoulli + occasional bursts
    def pulsed_spend(on_prob, mean_on, sd_on, max_spend=None):
        on = rng.random(n) < on_prob
        spend = np.where(on, rng.normal(mean_on, sd_on, size=n), 0.0)
        spend = np.clip(spend, 0, None)
        if max_spend is not None:
            spend = np.minimum(spend, max_spend)
        return spend

    # Tune these to resemble the scale
    tiktok = pulsed_spend(on_prob=0.30, mean_on=10500, sd_on=2500, max_spend=16000)
    facebook = pulsed_spend(on_prob=0.40, mean_on=4800, sd_on=1200, max_spend=9000)
    google = pulsed_spend(on_prob=0.60, mean_on=2050, sd_on=250, max_spend=3500)

    # If there is no campaign, spend is 0. Some occasional weeks with all channels off. 
    blackout = rng.random(n) < 0.06
    tiktok[blackout] = 0
    facebook[blackout] = 0
    google[blackout] = 0

    # (2) Transform spends: adstock + saturation
    # Carryover
    ad_tiktok = geometric_adstock(tiktok, alpha=0.55)
    ad_facebook = geometric_adstock(facebook, alpha=0.35)
    ad_google = geometric_adstock(google, alpha=0.25)

    # Diminishing returns (half-saturation roughly around typical adstock levels)
    sat_tiktok = hill_saturation(ad_tiktok, half_saturation=18000, slope=1.2)
    sat_facebook = hill_saturation(ad_facebook, half_saturation=8000, slope=1.2)
    sat_google = hill_saturation(ad_google, half_saturation=3500, slope=1.1)

    # (3) Baseline sales: trend + yearly seasonality
    base = 7800
    trend = 6.5 * t  # gentle growth over time
    yearly = 1200 * np.sin(2 * np.pi * t / 52.0) + 500 * np.cos(2 * np.pi * t / 52.0)

    # (4) Channel contributions
    # Scale contributions so Sales ends up around ~5k–18k
    contrib = (
        5200 * sat_tiktok +
        2600 * sat_facebook +
        3000 * sat_google
    )

    # (5) Noise
    noise = rng.normal(0, 700, size=n)  # adjust for more/less volatility

    sales = base + trend + yearly + contrib + noise
    sales = np.clip(sales, 0, None)

    df = pd.DataFrame({
        "Date": dates,
        "TikTok": np.round(tiktok, 2),
        "Facebook": np.round(facebook, 2),
        "Google Ads": np.round(google, 2),
        "Sales": np.round(sales, 2),
    })

    return df

# Example usage
df = simulate_mmm_like_data()
df.to_csv("synthetic_mmm_data.csv", index=False)
print(df.head(15))
# df.to_csv("synthetic_mmm_data.csv", index=False)
