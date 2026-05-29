# AI Bias Laboratory — Phase 1

Internal bias detection and fairness auditing platform for AI/ML models.

## Requirements

```bash
pip install -r requirements.txt
```

## Run

```bash
streamlit run app/main.py
```

Open browser at: `http://localhost:8501`

## Screens

| Screen | Path |
|--------|------|
| Dashboard | `app/main.py` |
| Bias Generator | `app/pages/02_bias_generator.py` |
| Benchmark Runner | `app/pages/03_benchmark_runner.py` |
| Fairness Metrics | `app/pages/04_fairness_metrics.py` |

## Project Structure

```
app/
├── main.py                    # Dashboard entry point
├── pages/
│   ├── 02_bias_generator.py   # Synthetic bias injection UI
│   ├── 03_benchmark_runner.py # BBQ / CrowS-Pairs / WinoBias
│   └── 04_fairness_metrics.py # Fairlearn metrics + amplification
└── core/
    ├── bias_generator.py      # inject_bias() engine
    ├── benchmarks.py          # Benchmark runners
    ├── metrics.py             # Fairness metrics calculation
    └── utils.py               # Seeding + data loading

data/sample_datasets/
└── hr_candidates.csv          # 1000-row synthetic HR dataset
```

## Demo Mode

Works out of the box — all pages auto-load the sample HR dataset.
No external API keys required.
