"""
Model 4: random forest regression.

Predicts the hip angle in degrees from the 40 EMG features of one 200 ms window.
The model averages 200 decision trees. Each tree is trained on a random sample of the windows and
needs at least 3 windows per leaf. Averaging many trees lets the model capture non-linear
relationships that ridge cannot. The number of trees barely changes the accuracy beyond about 100.
The angle is clipped to 0-50 degrees and floored into one of 10 states of 5 degrees.

The label, filtering, features and windowing come from src/emg_pipeline.py and the test protocol from
src/model_runner.py. This file lives in "Data Analysis 1 - Sara/".

Usage (run from the repository root)
  python "Data Analysis 1 - Sara/model4.py"                  cross-validate on the files in Data/
  python "Data Analysis 1 - Sara/model4.py" --sanity         also run the shifted-label check
  python "Data Analysis 1 - Sara/model4.py" --test-stand A.xlsx --test-walk B.xlsx
                                                             train on Data/, test on a new recording
Predictions are saved in a results/ folder next to this file.
"""
import sys
from pathlib import Path

from sklearn.ensemble import RandomForestRegressor

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import model_runner


def build_model():
    return RandomForestRegressor(n_estimators=200, min_samples_leaf=3, random_state=0)


if __name__ == "__main__":
    model_runner.run_cli("model4", build_model, __file__)
