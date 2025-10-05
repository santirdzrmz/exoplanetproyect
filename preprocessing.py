from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
import numpy as np
from lightkurve.periodogram import BoxLeastSquares
positive=["CP", "PC", "KP"]

def curves(name, i, label, target_len=500, phase_window=0.1):
  search = lk.search_lightcurve('TIC '+str(name), mission="TESS", author="SPOC")
  if len(search) == 0:
      return [], []
  print("........ "+ str(i+1))
  lc = search.download()
  lc = lc.remove_nans().normalize()

  # Quitar tendencia instrumental
  trend = savgol_filter(lc.flux.value, window_length=301, polyorder=2)
  flux_flat = lc.flux.value / trend

  t, f = lc.time.value, flux_flat
  # Detección de periodo
  bls = BoxLeastSquares(t, f)
  periods = np.linspace(0.5, 20, 4000)
  result = bls.power(periods, 0.05)
  best = np.argmax(result.power)
  period, t0 = result.period[best], result.transit_time[best]

  # Plegar la curva en fase centrando tránsito
  phase = ((t - t0 + 0.5 * period) % period) / period - 0.5
  mask = (phase > -phase_window) & (phase < phase_window)
  phase, f = phase[mask], f[mask]

  # Ordenar por fase y normalizar
  order = np.argsort(phase)
  phase, f = phase[order], f[order]
  f = (f - np.median(f)) / np.std(f)
  if len(f) > target_len:
      f = f[:target_len]
  else:
      f = np.pad(f, (0, target_len - len(f)), "constant", constant_values=0)

  label = 1 if label in positive else 0
  return f.astype(np.float32), label
