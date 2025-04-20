# Bias-Variance Trade-off with Knots
A minimal simulation to visualize the bias-variance trade-off in knot-based smoothing of noisy time series data using linear interpolation. Includes an interactive Plotly chart to compare approximations with varying numbers of knots.

## Setup
```bash
poetry install
poetry run python knot_bias_variance.py
```

This will:
- Generate W_<n>knots.csv files
- Output Bias-Variance Trade-off in Number of Knots.html

## Files
- knot_bias_variance.py: Main script
- *.csv: Weighting matrices
- *.html: Interactive visualization

## Dependencies
Managed via Poetry. See pyproject.toml.
