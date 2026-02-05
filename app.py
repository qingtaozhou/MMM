from flask import Flask, Response, render_template_string
import io

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # important for servers (no GUI)
import matplotlib.pyplot as plt

# -------------------------------------------------------
# Load the outputs created by your MMM script
# (or replace this section with direct model execution)
# -------------------------------------------------------

summary = pd.read_csv("mmm_roas_summary.csv")  # columns: Channel, Total Spend, Total Incremental Sales, ROAS
out = pd.read_csv("mmm_contributions.csv")
out["Date"] = pd.to_datetime(out["Date"])

# Ensure expected columns exist
required_summary_cols = {"Channel", "Total Spend", "Total Incremental Sales", "ROAS"}
required_out_cols = {"Date", "Sales", "TikTok_incremental_sales", "Facebook_incremental_sales", "Google Ads_incremental_sales"}

missing_s = required_summary_cols - set(summary.columns)
missing_o = required_out_cols - set(out.columns)

if missing_s:
    raise ValueError(f"summary missing columns: {missing_s}")
if missing_o:
    raise ValueError(f"out missing columns: {missing_o}")

app = Flask(__name__)

# -------------------------------------------------------
# Helper: return a matplotlib figure as PNG response
# -------------------------------------------------------
def fig_to_png_response(fig) -> Response:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return Response(buf.getvalue(), mimetype="image/png")

# -------------------------------------------------------
# Routes
# -------------------------------------------------------
@app.route("/")
def index():
    # Simple HTML page that embeds the images
    html = """
    <!doctype html>
    <html>
      <head>
        <title>MMM Dashboard</title>
        <style>
          body { font-family: Arial, sans-serif; margin: 24px; }
          .card { border: 1px solid #ddd; border-radius: 12px; padding: 16px; margin-bottom: 18px; }
          img { max-width: 100%; height: auto; }
          .row { display: flex; gap: 18px; flex-wrap: wrap; }
          .half { flex: 1 1 450px; }
        </style>
      </head>
      <body>
        <h1>MMM Figures</h1>

        <div class="row">
          <div class="card half">
            <h2>ROAS by Channel</h2>
            <img src="/plot/roas.png" alt="ROAS Plot">
          </div>

          <div class="card half">
            <h2>Spend vs Incremental Sales</h2>
            <img src="/plot/scale.png" alt="Scale Plot">
          </div>
        </div>

        <div class="card">
          <h2>Weekly Incremental Sales by Channel</h2>
          <img src="/plot/contributions.png" alt="Contributions Plot">
        </div>

        <div class="card">
          <h2>Stacked Incremental Sales</h2>
          <img src="/plot/stacked.png" alt="Stacked Plot">
        </div>

        <div class="card">
          <h2>Observed Sales vs Total Incremental</h2>
          <img src="/plot/decomposition.png" alt="Decomposition Plot">
        </div>
      </body>
    </html>
    """
    return render_template_string(html)

@app.route("/plot/roas.png")
def plot_roas():
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.bar(summary["Channel"], summary["ROAS"])
    ax.set_title("Channel ROAS (Incremental Sales / Spend)")
    ax.set_ylabel("ROAS")
    ax.axhline(1.0, linestyle="--", linewidth=1)
    return fig_to_png_response(fig)

@app.route("/plot/scale.png")
def plot_scale():
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.scatter(summary["Total Spend"], summary["Total Incremental Sales"])
    for _, r in summary.iterrows():
        ax.text(r["Total Spend"], r["Total Incremental Sales"], r["Channel"], ha="left", va="bottom")
    ax.set_title("Channel Scale vs Incremental Impact")
    ax.set_xlabel("Total Spend")
    ax.set_ylabel("Total Incremental Sales")
    return fig_to_png_response(fig)

@app.route("/plot/contributions.png")
def plot_contributions():
    fig, ax = plt.subplots(figsize=(12, 5.5))
    ax.plot(out["Date"], out["TikTok_incremental_sales"], label="TikTok")
    ax.plot(out["Date"], out["Facebook_incremental_sales"], label="Facebook")
    ax.plot(out["Date"], out["Google Ads_incremental_sales"], label="Google Ads")
    ax.set_title("Weekly Incremental Sales by Channel (MMM)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Incremental Sales")
    ax.legend()
    return fig_to_png_response(fig)

@app.route("/plot/stacked.png")
def plot_stacked():
    fig, ax = plt.subplots(figsize=(12, 5.5))
    ax.stackplot(
        out["Date"],
        out["TikTok_incremental_sales"],
        out["Facebook_incremental_sales"],
        out["Google Ads_incremental_sales"],
        labels=["TikTok", "Facebook", "Google Ads"]
    )
    ax.set_title("Stacked Weekly Incremental Sales by Channel")
    ax.set_xlabel("Date")
    ax.set_ylabel("Incremental Sales")
    ax.legend(loc="upper left")
    return fig_to_png_response(fig)

@app.route("/plot/decomposition.png")
def plot_decomposition():
    fig, ax = plt.subplots(figsize=(12, 5.5))
    total_incr = (
        out["TikTok_incremental_sales"]
        + out["Facebook_incremental_sales"]
        + out["Google Ads_incremental_sales"]
    )
    ax.plot(out["Date"], out["Sales"], label="Observed Sales")
    ax.plot(out["Date"], total_incr, label="Total Incremental (MMM)")
    ax.set_title("Observed Sales vs Media-Driven Incremental Sales")
    ax.set_xlabel("Date")
    ax.set_ylabel("Sales")
    ax.legend()
    return fig_to_png_response(fig)

if __name__ == "__main__":
    # For local dev:
    app.run(host="0.0.0.0", port=8000, debug=True)
