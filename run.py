
from mmm.data import MMMData
from mmm.model import BayesianMMM


def main():
    channels = ["TikTok", "Facebook", "Google Ads"]
    data = MMMData.from_csv("./mmm/synthetic_mmm_data.csv", channels)

    train_idx, _ = data.split(test_frac=0.2)

    mmm = BayesianMMM(data, train_idx=train_idx)
    mmm.build_model()
    mmm.fit()

    out, summary = mmm.compute_contributions_and_roas()
    print(summary.to_string(index=False))

    rmse, mae = mmm.holdout_eval(test_frac=0.2)
    print("\nHoldout RMSE:", rmse)
    print("Holdout MAE :", mae)

    out.to_csv("mmm_contributions.csv", index=False)
    summary.to_csv("mmm_roas_summary.csv", index=False)


if __name__ == "__main__":
    main()
