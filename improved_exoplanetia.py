import numpy as np
import pandas as pd
import lightkurve as lk
from astropy.timeseries import BoxLeastSquares
from scipy.signal import savgol_filter
import concurrent.futures
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
import matplotlib.pyplot as plt


positive = ["CP"]
negative = ["FP"]  # Confirmed planet labels
TARGET_LEN = 200               # phase-binned points per curve
PHASE_WINDOW = 0.1             # phase window around transit


def curves(name, label, target_len=TARGET_LEN, phase_window=PHASE_WINDOW):

    search = lk.search_lightcurve(f"TIC {name}", mission="TESS", author="SPOC")
    if len(search) == 0:
        return [], []

    lc = search.download(download_dir="lightcurves_cache")
    lc = lc.remove_nans().normalize()

    # Remove slow instrumental trends
    trend = savgol_filter(lc.flux.value, 301, 2)
    flux_flat = lc.flux.value / trend
    t, f = lc.time.value, flux_flat

    # --- Detect period via Box Least Squares
    bls = BoxLeastSquares(t, f)
    periods = np.linspace(0.5, 20, 10000)
    result = bls.power(periods, 0.05)
    best = np.argmax(result.power)
    period, t0 = result.period[best], result.transit_time[best]

    # --- Phase folding around transit
    phase = ((t - t0 + 0.5 * period) % period) / period - 0.5
    mask = (phase > -phase_window) & (phase < phase_window)
    phase, f = phase[mask], f[mask]

    # --- Sort and normalize
    order = np.argsort(phase)
    f = f[order]
    f = (f - np.median(f)) / np.std(f)

    # --- Center the transit at mid-point automatically
    mid = np.argmin(f)
    shift = len(f)//2 - mid
    f = np.roll(f, shift)

    # --- Phase binning (smooth and equal length)
    bins = target_len
    binned = np.array_split(f, bins)
    f = np.array([np.mean(b) for b in binned])

    # --- Pad/truncate to fixed size (should already match target_len)
    if len(f) > target_len:
        f = f[:target_len]
    else:
        f = np.pad(f, (0, target_len - len(f)), 'constant', constant_values=0.0)
    if label in positive:
       label_bin=1
    elif label in negative:
       label_bin=0
    return f.astype(np.float32), label_bin



import pandas as pd # type: ignore
df = pd.read_csv("C:\\Users\\santi\\Downloads\\TOI_2025.10.04_12.31.14.csv")
to_keep=["tid", "tfopwg_disp"]
counter=0

for column in df.columns.tolist():
  if column not in to_keep:
    df.drop(column, axis=1, inplace=True)

df_yes=df[df["tfopwg_disp"].isin(positive)]
df_no=df[df["tfopwg_disp"].isin(negative)]
df_yes=df_yes.reset_index(drop=True)
df_no=df_no.reset_index(drop=True)

X = []
y = []

for i in range(500):
  curve, label = curves(df_yes["tid"][i], df_yes["tfopwg_disp"][i])
  print(str(i+1)+" CP")
  if len(curve) == 0:
    continue
  X.append(curve)
  y.append(label)

for i in range(500):
  curve, label = curves(df_no["tid"][i], df_no["tfopwg_disp"][i])
  print(str(i+1)+" NP")

  if len(curve) == 0:
    continue
  X.append(curve)
  y.append(label)

X = np.array(X, dtype=np.float32)

y = np.array(y, dtype=np.float32)


def build_model(X):
    model = models.Sequential([
        layers.Conv1D(64, 7, activation='relu', input_shape=(X.shape[1], 1)),
        layers.MaxPooling1D(2),
        layers.Conv1D(128, 5, activation='relu'),
        layers.MaxPooling1D(2),
        layers.GlobalAveragePooling1D(),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model
print(X)

def train_exoplanet_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

    model = build_model(X)
    callback = tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)

    history = model.fit(
        X_train, y_train,
        epochs=80,
        batch_size=8,
        validation_data=(X_test, y_test),
        callbacks=[callback],
        class_weight={0:1, 1:1},
        verbose=1
    )

    # Evaluation
    y_pred = (model.predict(X_test) > 0.5).astype("int32")
    print("\n📊 Classification Report:")
    print(classification_report(y_test, y_pred))
    print("AUC:", roc_auc_score(y_test, y_pred))

    # Training plots
    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    plt.plot(history.history['accuracy'], label='Train')
    plt.plot(history.history['val_accuracy'], label='Val')
    plt.title('Accuracy')
    plt.legend()

    plt.subplot(1,2,2)
    plt.plot(history.history['loss'], label='Train')
    plt.plot(history.history['val_loss'], label='Val')
    plt.title('Loss')
    plt.legend()
    plt.show()

    model.save("exoplanetia_model.keras")
    print("✅ Model saved as exoplanetia_model.keras")

train_exoplanet_model(X, y)
