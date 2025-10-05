import numpy as np # type: ignore
from scipy.stats import norm # type: ignore
import tensorflow as tf # type: ignore
import lightkurve as lk # type: ignore
import matplotlib as plt # type: ignore
from astropy.timeseries import BoxLeastSquares # type: ignore
from scipy.signal import savgol_filter # type: ignore

positive = ["CP"]
negative = ["FP"]  

TARGET_LEN = 200               
PHASE_WINDOW = 0.1          

def curves(name, label, target_len=TARGET_LEN, phase_window=PHASE_WINDOW):

    search = lk.search_lightcurve(f"TIC {name}", mission="TESS", author="SPOC")
    if len(search) == 0:
        return [], []

    lc = search.download(download_dir="lightcurves_cache")
    lc = lc.remove_nans().normalize()

    trend = savgol_filter(lc.flux.value, 301, 2)
    flux_flat = lc.flux.value / trend
    t, f = lc.time.value, flux_flat

    bls = BoxLeastSquares(t, f)
    periods = np.linspace(0.5, 20, 10000)
    result = bls.power(periods, 0.05)
    best = np.argmax(result.power)
    period, t0 = result.period[best], result.transit_time[best]

    phase = ((t - t0 + 0.5 * period) % period) / period - 0.5
    mask = (phase > -phase_window) & (phase < phase_window)
    phase, f = phase[mask], f[mask]

    order = np.argsort(phase)
    f = f[order]
    f = (f - np.median(f)) / np.std(f)

    mid = np.argmin(f)
    shift = len(f)//2 - mid
    f = np.roll(f, shift)

    bins = target_len
    binned = np.array_split(f, bins)
    f = np.array([np.mean(b) for b in binned])

    if len(f) > target_len:
        f = f[:target_len]
    else:
        f = np.pad(f, (0, target_len - len(f)), 'constant', constant_values=0.0)
    if label in positive:
       label_bin=1
    elif label in negative:
       label_bin=0
    return f.astype(np.float32), label_bin

def ensure_dropout(model):
    return model

def mc_dropout_proba(model, X, T=100, batch_size=None):
    probs = []
    for _ in range(T):
        p = model(X, training=True).numpy().ravel()  
        probs.append(p[0])
    probs = np.array(probs)
    return probs

def ci_mean_proba(probs, alpha=0.05):
    pbar = probs.mean()
    se = probs.std(ddof=1) / np.sqrt(len(probs))
    z = norm.ppf(1 - alpha/2)
    lo, hi = pbar - z*se, pbar + z*se
    return float(np.clip(lo, 0, 1)), float(np.clip(hi, 0, 1)), float(pbar), float(se)


def predict_with_uncertainty(model, X_input, T=200, threshold=0.5):
    probs = mc_dropout_proba(model, X_input, T=T)
    lo_p, hi_p, pbar, se = ci_mean_proba(probs, alpha=0.05)
    return pbar, [lo_p, hi_p]

model = tf.keras.models.load_model("exoplanetia_model.keras")

T = 50
thr = 0.1

# INTERFAZ
name=input("Tess Input Catalog (TIC), solo número: ") #CAMBIAR POR INTERFAZ
X_input, _=curves(name, "FP")
X = np.array(X_input, dtype=np.float32).reshape(1, 200, 1)


prob, interval = predict_with_uncertainty(model, X, T=T, threshold=thr)
prob_max=0.4
print("Probabilidad de Exoplaneta: " +str(prob)) #CAMBIAR POR INTERFAZ
print("Intervalo de Confianza: ", interval) #CAMBIAR POR INTERFAZ



