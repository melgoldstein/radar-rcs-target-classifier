# Coastal radar return classification

[Portfolio: melgoldstein.com](https://melgoldstein.com/)

Synthetic X-band-inspired return features for Port Canaveral (28.4158 N, 80.6048 W).
Illustrative seabird, quadcopter and fixed-wing distributions overlap; these are not measured signatures.

Run `python -m pip install -r requirements.txt`, `python -m src.generate_returns`,
then `python -m src.train_classifier`. Outputs include test precision/recall,
confusion matrices and permutation feature importances under `artifacts/`.

Flights, not individual observations, are partitioned into train and test cohorts.
RandomForest and GradientBoosting use fixed seeds. Labels and flight IDs are excluded
from predictors. Permutation importance is computed on the held-out cohort.
Check: `python -m unittest discover -s tests`.
No field data, operational target identification or sensor-specific claims are included.
