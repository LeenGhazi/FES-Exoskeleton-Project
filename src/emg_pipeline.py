"""
What it does
  1. Cleans both recordings (drops the NaN tail, interpolates gaps).
  2. Estimates a hip flexion angle from the THIGH IMU (label for the regressor).
  3. Filters the 8 EMG channels and extracts 5 time-domain features per channel (40 total).
  4. Trains baseline regressors, bins predictions into 10 states, reports honest metrics.

"""
import sys
import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt, iirnotch, filtfilt
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# ----------------------------------------------------------------- config
FS = 2000                  # Hz, from Time_s
IMU_FS = 200               # Hz, native IMU rate (the sheet is interpolated up to 2 kHz)
LEG = "R"                  # which thigh IMU gives the label: "L" or "R"
SIGN = None                # ASSUMPTION: None = auto (flexion = the larger excursion). Set +1 / -1 once confirmed.
WIN_MS, STEP_MS = 200, 50  # EMG analysis window and hop
LOOKAHEAD_MS = 100         # predict the angle this far after the window (intent precedes motion)
BIN_DEG, MAX_DEG = 5, 50   # 10 states: 0-5 -> 0, 5-10 -> 1, ... 45-50 -> 9
N_STATES = MAX_DEG // BIN_DEG
MAINS_HZ = 60              # Saudi Arabia
EMG_COLS = ["Multifidus_R", "Multifidus_L", "ExternalOblique_R", "ExternalOblique_L",
            "RectusAbdominis_R", "RectusAbdominis_L", "ErectorSpinae_R", "ErectorSpinae_L"]


# ----------------------------------------------------------------- loading
def load(path):
    df = pd.read_excel(path)
    imu = [c for c in df.columns if c != "Time_s" and c not in EMG_COLS]
    # The last ~10 rows are NaN in every IMU column (export artifact): trim them.
    df = df[df[imu].notna().any(axis=1)].reset_index(drop=True) if df[imu].isna().all(axis=1).any() else df
    df = df.iloc[: df[[f"Thigh_{LEG}_Accel_X"]].last_valid_index() + 1]
    return df.interpolate(limit_direction="both").reset_index(drop=True)


# ----------------------------------------------------------------- hip angle label
def thigh_cols(leg):
    return ([f"Thigh_{leg}_Accel_{k}" for k in "XYZ"], [f"Thigh_{leg}_Gyro_{k}" for k in "XYZ"])


def fit_axis(standing, walking, leg):
    """Rest gravity direction (thigh long axis) and the swing axis perpendicular to it."""
    A, G = thigh_cols(leg)
    g0 = standing[A].mean().values
    g0u = g0 / np.linalg.norm(g0)
    bias = standing[G].median().values
    gy = walking[G].values - bias
    gy_plane = gy - np.outer(gy @ g0u, g0u)               # keep rotation perpendicular to the thigh
    _, _, vt = np.linalg.svd(gy_plane - gy_plane.mean(0), full_matrices=False)
    return g0u, bias, vt[0]


def hip_angle(df, leg, g0u, bias, n):
    """Complementary filter: gyro for dynamics, accelerometer (gravity) to stop drift."""
    A, G = thigh_cols(leg)
    sos = butter(2, 6, "low", fs=FS, output="sos")
    acc = sosfiltfilt(sos, df[A].values, axis=0)
    e1 = g0u - (g0u @ n) * n
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(n, e1)
    acc_ang = np.degrees(np.arctan2(acc @ e2, acc @ e1))
    gz = (df[G].values - bias) @ n
    alpha, ang = 0.995, np.empty(len(acc_ang))
    ang[0] = acc_ang[0]
    for i in range(1, len(ang)):
        ang[i] = alpha * (ang[i - 1] + gz[i] / FS) + (1 - alpha) * acc_ang[i]
    return ang


# ----------------------------------------------------------------- EMG
def filter_emg(x):
    sos = butter(4, [20, 450], "bandpass", fs=FS, output="sos")
    b, a = iirnotch(MAINS_HZ, Q=30, fs=FS)
    return filtfilt(b, a, sosfiltfilt(sos, x, axis=0), axis=0)


def features(w, thr):
    """w: (n_samples, n_channels) window, already baseline-normalised. Returns 5 features per channel."""
    mav = np.mean(np.abs(w), axis=0)
    rms = np.sqrt(np.mean(w ** 2, axis=0))
    wl = np.sum(np.abs(np.diff(w, axis=0)), axis=0)
    zc = np.sum((w[:-1] * w[1:] < 0) & (np.abs(w[:-1] - w[1:]) > thr), axis=0)
    d = np.diff(w, axis=0)
    ssc = np.sum((d[:-1] * d[1:] < 0) & ((np.abs(d[:-1]) > thr) | (np.abs(d[1:]) > thr)), axis=0)
    return np.concatenate([mav, rms, wl, zc, ssc])


FEATURE_NAMES = [f"{f}_{c}" for f in ["MAV", "RMS", "WL", "ZC", "SSC"] for c in EMG_COLS]


def build(df, angle, rms_rest, name):
    emg = filter_emg(df[EMG_COLS].values) / rms_rest      # 1.0 = resting RMS of that channel
    win, step, look = FS * WIN_MS // 1000, FS * STEP_MS // 1000, FS * LOOKAHEAD_MS // 1000
    rows, y, t = [], [], []
    for s in range(0, len(emg) - win - look, step):
        rows.append(features(emg[s:s + win], thr=0.1))
        y.append(angle[s + win + look])
        t.append(s / FS)
    X = pd.DataFrame(rows, columns=FEATURE_NAMES)
    X["recording"], X["t_start_s"], X["hip_deg"] = name, t, np.clip(y, 0, MAX_DEG)
    return X


# ----------------------------------------------------------------- states
def to_state(deg):
    return np.clip(np.floor(np.asarray(deg) / BIN_DEG), 0, N_STATES - 1).astype(int)


# ----------------------------------------------------------------- evaluation
def evaluate(data, n_folds=4):
    """Contiguous (not shuffled) folds over the walking trial; standing windows always stay in training.
    A purge gap around each test fold removes windows that overlap it in time."""
    walk = data[data.recording == "walking"].reset_index(drop=True)
    stand = data[data.recording == "standing"]
    gap = (WIN_MS + LOOKAHEAD_MS) / STEP_MS
    edges = np.linspace(0, len(walk), n_folds + 1).astype(int)
    models = {
        "mean baseline": DummyRegressor(),
        "ridge": make_pipeline(StandardScaler(), Ridge(alpha=10.0)),
        "random forest": RandomForestRegressor(n_estimators=200, min_samples_leaf=3, random_state=0),
    }
    out = {k: np.full(len(walk), np.nan) for k in models}
    for f in range(n_folds):
        te = np.arange(edges[f], edges[f + 1])
        tr = np.array([i for i in range(len(walk)) if i < edges[f] - gap or i >= edges[f + 1] + gap])
        Xtr = pd.concat([walk.loc[tr, FEATURE_NAMES], stand[FEATURE_NAMES]])
        ytr = np.concatenate([walk.loc[tr, "hip_deg"], stand["hip_deg"]])
        for k, m in models.items():
            out[k][te] = m.fit(Xtr, ytr).predict(walk.loc[te, FEATURE_NAMES])
    y = walk.hip_deg.values
    print(f"\n{'model':15s} {'MAE(deg)':>9s} {'R2':>7s} {'exact state':>12s} {'within +/-1':>12s}")
    for k, p in out.items():
        p = np.clip(p, 0, MAX_DEG)
        ds = np.abs(to_state(p) - to_state(y))
        print(f"{k:15s} {mean_absolute_error(y, p):9.2f} {r2_score(y, p):7.2f} {np.mean(ds == 0):12.2f} {np.mean(ds <= 1):12.2f}")
    return walk, out


# ----------------------------------------------------------------- main
if __name__ == "__main__":
    stand_path, walk_path = (sys.argv[1:3] if len(sys.argv) >= 3 else
                             ["standing_still_z.xlsx", "Walking_z_1.xlsx"])
    stand, walk = load(stand_path), load(walk_path)

    g0u, bias, n = fit_axis(stand, walk, LEG)
    ang_w = hip_angle(walk, LEG, g0u, bias, n)
    ang_s = hip_angle(stand, LEG, g0u, bias, n)
    sign = SIGN or (1 if abs(ang_w.max()) >= abs(ang_w.min()) else -1)   # ASSUMPTION (see config)
    ang_w, ang_s = sign * ang_w, sign * ang_s
    print(f"leg={LEG}  swing axis (sensor frame)={n.round(2)}  sign={sign:+d}")
    print(f"walking hip angle: min {ang_w.min():.1f}  max {ang_w.max():.1f} deg   (standing: {ang_s.min():.1f} to {ang_s.max():.1f})")

    rms_rest = filter_emg(stand[EMG_COLS].values).std(axis=0)   # per-channel baseline from the standing file
    data = pd.concat([build(stand, ang_s, rms_rest, "standing"), build(walk, ang_w, rms_rest, "walking")],
                     ignore_index=True)
    data["state"] = to_state(data.hip_deg)
    data.to_csv("features_and_labels.csv", index=False)
    print(f"{len(data)} windows saved to features_and_labels.csv; state counts:\n"
          f"{data.groupby('recording').state.value_counts().sort_index().to_string()}")

    evaluate(data)
