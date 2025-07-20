import numpy as np
import sounddevice as sd
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import minimize
import time

SAMPLERATE = 44100
BUFFER_SIZE = 2048
CHANNEL = 0
audio_buffer = np.zeros(BUFFER_SIZE)

def audio_callback(indata, frames, time_info, status):
    global audio_buffer
    audio_buffer = indata[:, CHANNEL]

def lf_model(t, E1, a, w, E2, b, Te, Tc):
    u = np.zeros_like(t)
    for i, ti in enumerate(t):
        if ti < Te:
            u[i] = E1 * np.exp(a * ti) * np.sin(w * ti)
        elif Te <= ti < Tc:
            u[i] = -E2 * (np.exp(-b * (ti - Te)) - np.exp(-b * (Tc - Te)))
    return u

def lf_loss(params, t, y):
    E1, a, w, E2, b, Te, Tc = params
    y_pred = lf_model(t, E1, a, w, E2, b, Te, Tc)
    return np.sum((y - y_pred) ** 2)

def fit_lf_model(signal, fs):
    t = np.linspace(0, len(signal) / fs, len(signal))
    init = [1.0, -100.0, 500.0, 0.5, 50.0, 0.002, 0.004]
    bounds = [
        (0.1, 5), (-1000, -1), (50, 2000),
        (0.1, 5), (1, 1000), (0.0005, 0.01), (0.001, 0.02)
    ]
    res = minimize(lf_loss, init, args=(t, signal), bounds=bounds, method='L-BFGS-B')
    if res.success:
        E1, a, w, E2, b, Te, Tc = res.x
        T0 = t[-1]
        Tp = np.pi / w
        Oq = Te / T0
        Rq = (Tc - Te) / T0
        alpha_m = Tp / Te if Te != 0 else 0
        return Oq, Rq, alpha_m
    else:
        return None, None, None

def extract_parameters(buffer, fs):
    deg = gaussian_filter1d(np.diff(buffer), sigma=2)
    peaks, _ = find_peaks(-deg, distance=int(fs / 800), prominence=0.01)
    for i in range(len(peaks) - 1):
        start, end = peaks[i], peaks[i + 1]
        cycle = deg[start:end:2]  # 軽量化のため間引き
        if len(cycle) < 100:
            continue
        oq, rq, alpha = fit_lf_model(cycle, fs)
        if oq and 0 < oq < 1 and 0 < rq < 1 and 0 < alpha < 2:
            print(f"Oq: {oq:.3f}, Rq: {rq:.3f}, αm: {alpha:.3f}")
            break  # 1周期分だけ表示で十分

# === メイン処理 ===
print("リアルタイムEGG解析（ARX-LFモデル）開始...")
stream = sd.InputStream(callback=audio_callback, channels=2, samplerate=SAMPLERATE, blocksize=BUFFER_SIZE)
with stream:
    while True:
        try:
            extract_parameters(audio_buffer.copy(), SAMPLERATE)
            time.sleep(0.5)
        except KeyboardInterrupt:
            print("終了しました。")
            break
