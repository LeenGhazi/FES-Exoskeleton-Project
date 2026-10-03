"""
Model 3: ridge regression.

Predicts the hip angle in degrees as a weighted sum of the 40 EMG features of one 200 ms window
(MAV, RMS, waveform length, zero crossings and slope sign changes of the 8 channels).
The features are standardized first, and a penalty (alpha = 10) keeps the weights from growing too
large, which helps because the features are strongly correlated and the training set is small.
The angle is clipped to 0-50 degrees and floored into one of 10 states of 5 degrees.
The model is tiny (40 weights and one offset), so it is easy to run on the ESP32.

The label, filtering, features and windowing come from src/emg_pipeline.py and the test protocol from
src/model_runner.py. This file lives in "Data Analysis 1 - Sara/".

Usage (run from the repository root)
  python "Data Analysis 1 - Sara/model3.py"                  cross-validate on the files in Data/
  python "Data Analysis 1 - Sara/model3.py" --sanity         also run the shifted-label check
  python "Data Analysis 1 - Sara/model3.py" --test-stand A.xlsx --test-walk B.xlsx
                                                             train on Data/, test on a new recording
Predictions are saved in a results/ folder next to this file.
"""
import sys
from pathlib import Path

from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import model_runner


def build_model():
    return make_pipeline(StandardScaler(), Ridge(alpha=10.0))


if __name__ == "__main__":
    model_runner.run_cli("model3", build_model, __file__)
