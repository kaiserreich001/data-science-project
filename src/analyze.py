"""Analyze the synthetic study-performance dataset and create portfolio figures."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "study_performance_synthetic.csv"
FIGURE_DIR = PROJECT_ROOT / "visualizations"
REPORT_DIR = PROJECT_ROOT / "reports"

FEATURES = [
    "weekly_study_hours",
    "attendance_rate_pct",
    "prior_gpa",
    "tutoring_access",
    "first_generation",
]
TARGET = "exam_score"
PALETTE = {"No": "#6C8EBF", "Yes": "#E07A5F"}


def load_data(path: Path = DATA_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} is missing. Run `python src/generate_data.py` first."
        )
    return pd.read_csv(path)


def fit_and_evaluate(data: pd.DataFrame) -> tuple[LinearRegression, dict[str, float], np.ndarray]:
    x = data[FEATURES]
    y = data[TARGET]
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.25, random_state=42
    )
    model = LinearRegression().fit(x_train, y_train)
    predictions = model.predict(x_test)

    metrics = {
        "test_mae": float(mean_absolute_error(y_test, predictions)),
        "test_r2": float(r2_score(y_test, predictions)),
        "study_score_pearson_r": float(
            data["weekly_study_hours"].corr(data[TARGET])
        ),
    }
    return model, metrics, predictions


def create_figures(data: pd.DataFrame, model: LinearRegression) -> None:
    sns.set_theme(style="whitegrid", context="notebook")
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    # Figure 1: distribution of exam scores.
    fig, ax = plt.subplots(figsize=(9, 5.5))
    sns.histplot(
        data,
        x="exam_score",
        bins=20,
        kde=True,
        color="#3D6B9A",
        edgecolor="white",
        ax=ax,
    )
    ax.axvline(data["exam_score"].mean(), color="#D1495B", linestyle="--", lw=2)
    ax.set(
        title="Distribution of Synthetic Exam Scores",
        xlabel="Exam score (out of 100)",
        ylabel="Number of students",
    )
    ax.text(
        data["exam_score"].mean() + 1.3,
        ax.get_ylim()[1] * 0.88,
        f"Mean = {data['exam_score'].mean():.1f}",
        color="#D1495B",
    )
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "score_distribution.png", dpi=180)
    fig.savefig(FIGURE_DIR / "score_distribution.svg")
    plt.close(fig)

    # Figure 2: study hours versus exam score, split by tutoring access.
    plot_data = data.assign(
        Tutoring=data["tutoring_access"].map({0: "No", 1: "Yes"})
    )
    fig, ax = plt.subplots(figsize=(9, 5.5))
    sns.scatterplot(
        data=plot_data,
        x="weekly_study_hours",
        y="exam_score",
        hue="Tutoring",
        palette=PALETTE,
        alpha=0.72,
        s=34,
        ax=ax,
    )
    sns.regplot(
        data=data,
        x="weekly_study_hours",
        y="exam_score",
        scatter=False,
        color="#1F2D3D",
        line_kws={"linewidth": 2},
        ax=ax,
    )
    ax.set(
        title="More Weekly Study Time Is Associated With Higher Exam Scores",
        xlabel="Weekly study hours",
        ylabel="Exam score (out of 100)",
        ylim=(50, 105),
    )
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "study_hours_vs_score.png", dpi=180)
    fig.savefig(FIGURE_DIR / "study_hours_vs_score.svg")
    plt.close(fig)

    # Figure 3: standardized regression coefficients.
    coefficients = pd.Series(model.coef_, index=FEATURES).sort_values()
    pretty_names = {
        "weekly_study_hours": "Weekly study hours",
        "attendance_rate_pct": "Attendance rate (pct)",
        "prior_gpa": "Prior GPA",
        "tutoring_access": "Tutoring access",
        "first_generation": "First generation",
    }
    fig, ax = plt.subplots(figsize=(9, 5.5))
    colors = ["#D1495B" if value < 0 else "#3D6B9A" for value in coefficients]
    ax.barh(
        [pretty_names[name] for name in coefficients.index],
        coefficients.values,
        color=colors,
    )
    ax.axvline(0, color="#333333", linewidth=1)
    ax.set(
        title="Regression Coefficients From the Held-Out Evaluation Model",
        xlabel="Change in exam score per one-unit increase",
        ylabel="",
    )
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "regression_coefficients.png", dpi=180)
    fig.savefig(FIGURE_DIR / "regression_coefficients.svg")
    plt.close(fig)


def build_summary(data: pd.DataFrame, model: LinearRegression, metrics: dict[str, float]) -> dict:
    tutoring_gap = (
        data.loc[data["tutoring_access"] == 1, TARGET].mean()
        - data.loc[data["tutoring_access"] == 0, TARGET].mean()
    )
    high_attendance = data["attendance_rate_pct"] >= 90
    attendance_gap = (
        data.loc[high_attendance, TARGET].mean()
        - data.loc[~high_attendance, TARGET].mean()
    )
    return {
        "records": int(len(data)),
        "average_exam_score": round(float(data[TARGET].mean()), 2),
        "average_weekly_study_hours": round(
            float(data["weekly_study_hours"].mean()), 2
        ),
        "study_hours_coefficient": round(
            float(model.coef_[FEATURES.index("weekly_study_hours")]), 2
        ),
        "tutoring_score_gap": round(float(tutoring_gap), 2),
        "attendance_score_gap_90pct": round(float(attendance_gap), 2),
        **{key: round(value, 3) for key, value in metrics.items()},
    }


def main() -> None:
    data = load_data()
    model, metrics, _ = fit_and_evaluate(data)
    create_figures(data, model)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    summary = build_summary(data, model, metrics)
    (REPORT_DIR / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
