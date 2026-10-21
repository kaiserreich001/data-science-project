# Study Habits and Academic Performance

A reproducible data-science project that examines how weekly study time, class attendance, prior GPA, tutoring access, and first-generation status relate to exam performance.

> **Important:** This project uses a **synthetic dataset**. It contains no real student records or personally identifiable information.

## Research question

To what extent are weekly study hours, attendance, prior GPA, tutoring access, and first-generation status associated with exam scores?

## Dataset source and usage permissions

The dataset is generated deterministically by [`src/generate_data.py`](src/generate_data.py) using `numpy.random.default_rng(7204)`. The simulation is released under **CC0 1.0 Universal**, meaning it may be copied, modified, and redistributed without attribution requirements. Because the data are synthetic, there are no consent, privacy, or restricted-dataset concerns. The resulting CSV is stored at [`data/study_performance_synthetic.csv`](data/study_performance_synthetic.csv).

## Methods and tools

- **Python** and **pandas** for data generation and descriptive analysis.
- **scikit-learn** for a linear regression model with a fixed 75/25 train-test split.
- **matplotlib** and **seaborn** for communication through charts.
- **Jupyter Notebook** for an executed, reproducible narrative.
- **GitHub Pages** for presenting the portfolio website.

The model estimates exam score from five explanatory variables. Performance is reported using test-set MAE and R², and associations are described without making causal claims.

## Key findings

- The mean exam score in the synthetic sample is **85.95**.
- Weekly study time has a positive association with exam score. The regression estimate is approximately **+1.43 points per additional weekly study hour**, holding the other variables constant.
- Students with tutoring access score about **4.48 points higher on average** than students without access.
- The correlation between weekly study hours and exam score is **0.651**.
- The held-out model has **MAE = 5.21** and **R² = 0.534**.

**One limitation:** The data are synthetic and were generated with a known relationship. The analysis therefore demonstrates a reproducible workflow and responsible interpretation rather than providing evidence about real students. The results are associations, not causal effects.

## Visualizations

### Score distribution

![Distribution of synthetic exam scores](visualizations/score_distribution.png)

### Study time and exam performance

![Study hours versus exam score](visualizations/study_hours_vs_score.png)

### Regression coefficients

![Regression coefficients](visualizations/regression_coefficients.png)

## Project structure

```text
data-science-project/
├── data/
│   └── study_performance_synthetic.csv
├── notebooks/
│   └── study_performance_analysis.ipynb
├── reports/
│   └── summary.json
├── src/
│   ├── analyze.py
│   └── generate_data.py
├── visualizations/
│   ├── regression_coefficients.png
│   ├── score_distribution.png
│   └── study_hours_vs_score.png
├── LICENSE
├── README.md
└── requirements.txt
```

## Run the analysis

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python src/generate_data.py
python src/analyze.py
jupyter nbconvert --to notebook --execute notebooks/study_performance_analysis.ipynb \
  --output study_performance_analysis.ipynb
```

The notebook and `src/analyze.py` use fixed random seeds, so the results and figures should be reproducible.

## License

The code, synthetic dataset, and visualizations are released under **CC0 1.0 Universal**. See [`LICENSE`](LICENSE).
