"""
Model 2: mean baseline.

Always predicts the average hip angle of the training windows, whatever the EMG says.
It is the reference the other models have to beat: a model that does not score better than this
has learned nothing from the EMG.

The label, filtering, features and windowing come from src/emg_pipeline.py and the test protocol from
src/model_runner.py.

Usage (run from the repository root)
  python "Data Analysis 1 - Sara/model2.py"                  cross-validate on the files in Data/
  python "Data Analysis 1 - Sara/model2.py" --sanity         also run the shifted-label check
  python "Data Analysis 1 - Sara/model2.py" --test-stand A.xlsx --test-walk B.xlsx
                                                             train on Data/, test on a new recording
Predictions are saved in a results/ folder next to this file.
"""
import sys
from pathlib import Path

from sklearn.dummy import DummyRegressor

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import model_runner


def build_model():
    return DummyRegressor()


if __name__ == "__main__":
    model_runner.run_cli("model2", build_model, __file__)
