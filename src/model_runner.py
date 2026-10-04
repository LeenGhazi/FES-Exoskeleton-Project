"""
Shared test harness for the single-model files in "Data Analysis 1 - Sara/" (model2.py, model3.py, model4.py).

A model file only defines build_model() and calls run_cli(). Everything else is here, so the three
files use exactly the same data, windows, labels and test protocol:
  - cross-validation: 4 contiguous folds over the walking trial, windows overlapping the test fold
    removed from training, standing windows always in training
  - shifted-label check: the same test with the walking labels rolled by half the trial
  - new recording: train on one standing and walking pair, test on the walking windows of another pair
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import emg_pipeline as P
import models_benchmark as B

FEATURES = list(P.FEATURE_NAMES)


def angles_to_states(angles):
    return P.to_state(np.clip(angles, 0, P.MAX_DEG))


def cross_validate(build_model, stand_path, walk_path, shift_labels=False):
    """Returns the walking windows and the out-of-fold predicted angles."""
    data = B.make_dataset(stand_path, walk_path)
    if shift_labels:
        walking = data.recording == "walking"
        data.loc[walking, "hip_deg"] = np.roll(data.loc[walking, "hip_deg"].values, walking.sum() // 2)
        data["state"] = P.to_state(data.hip_deg)
    return B.cv_predict(data, FEATURES, build_model, "reg")


def test_on_new_recording(build_model, train_stand, train_walk, test_stand, test_walk):
    """Fits on every window of the training pair and predicts the walking windows of the new pair."""
    train = B.make_dataset(train_stand, train_walk)
    test = B.make_dataset(test_stand, test_walk)
    test = test[test.recording == "walking"].reset_index(drop=True)
    model = build_model().fit(train[FEATURES], train.hip_deg)
    return test, model.predict(test[FEATURES])


def report(title, windows, pred_deg):
    s = B.score(windows.hip_deg.values, pred_deg)
    print(f"{title}: MAE {s['MAE_deg']:.2f} deg | R2 {s['R2']:.2f} | corr {s['corr']:.2f} | "
          f"exact state {s['exact']:.0%} | within +/-1 state {s['within1']:.0%}")
    true_state = pd.Series(windows.state.values, name="true state")
    pred_state = pd.Series(angles_to_states(pred_deg), name="predicted state")
    print(pd.crosstab(true_state, pred_state).to_string())


def save_predictions(path, windows, pred_deg):
    out = pd.DataFrame({"t_s": windows.t_start_s, "true_deg": windows.hip_deg, "pred_deg": pred_deg,
                        "true_state": windows.state, "pred_state": angles_to_states(pred_deg)})
    path.parent.mkdir(exist_ok=True)
    out.to_csv(path, index=False)


def run_cli(model_name, build_model, model_file):
    """Command line entry point. model_file is the __file__ of the model script."""
    here = Path(model_file).resolve().parent
    root = here.parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-stand", default=root / "Data" / "standing still z.xlsx")
    parser.add_argument("--train-walk", default=root / "Data" / "Walking z 1.xlsx")
    parser.add_argument("--test-stand")
    parser.add_argument("--test-walk")
    parser.add_argument("--sanity", action="store_true")
    args = parser.parse_args()

    if args.test_stand and args.test_walk:
        windows, pred = test_on_new_recording(build_model, args.train_stand, args.train_walk,
                                              args.test_stand, args.test_walk)
        report(f"{model_name}, new recording", windows, pred)
        save_predictions(here / "results" / f"{model_name}_new_recording.csv", windows, pred)
    else:
        windows, pred = cross_validate(build_model, args.train_stand, args.train_walk)
        report(f"{model_name}, cross-validation", windows, pred)
        save_predictions(here / "results" / f"{model_name}_cv.csv", windows, pred)
        if args.sanity:
            shifted_windows, shifted_pred = cross_validate(build_model, args.train_stand, args.train_walk,
                                                           shift_labels=True)
            print()
            report(f"{model_name}, shifted labels (should be clearly worse)", shifted_windows, shifted_pred)
