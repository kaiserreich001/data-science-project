"""Generate a reproducible synthetic student-performance dataset.

The data are created for portfolio demonstration only. No real students or
personally identifiable information are represented.
"""

from pathlib import Path

import numpy as np
import pandas as pd


SEED = 7204
N_STUDENTS = 420


def generate_study_data(seed: int = SEED, n_students: int = N_STUDENTS) -> pd.DataFrame:
    """Create a synthetic dataset with realistic, non-perfect relationships."""
    rng = np.random.default_rng(seed)

    prior_gpa = np.clip(rng.normal(3.0, 0.45, n_students), 1.6, 4.0)
    study_hours = np.clip(rng.gamma(shape=5.0, scale=2.0, size=n_students), 0, 30)
    attendance_rate = np.clip(rng.beta(7.5, 2.2, n_students), 0.45, 1.0)
    tutoring_access = rng.binomial(1, 0.38, n_students)
    first_generation = rng.binomial(1, 0.42, n_students)

    noise = rng.normal(0, 6.0, n_students)
    exam_score = (
        30
        + 1.70 * study_hours
        + 28.0 * attendance_rate
        + 5.8 * prior_gpa
        + 4.2 * tutoring_access
        - 2.8 * first_generation
        + noise
    )
    exam_score = np.clip(exam_score, 0, 100)

    return pd.DataFrame(
        {
            "student_id": [f"S{i:04d}" for i in range(1, n_students + 1)],
            "weekly_study_hours": np.round(study_hours, 1),
            "attendance_rate_pct": np.round(attendance_rate * 100, 1),
            "prior_gpa": np.round(prior_gpa, 2),
            "tutoring_access": tutoring_access,
            "first_generation": first_generation,
            "exam_score": np.round(exam_score, 1),
        }
    )


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    output_path = project_root / "data" / "study_performance_synthetic.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data = generate_study_data()
    data.to_csv(output_path, index=False)
    print(f"Wrote {len(data)} rows to {output_path}")


if __name__ == "__main__":
    main()
