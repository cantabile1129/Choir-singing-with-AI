import numpy as np
import sounddevice as sd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.signal import find_peaks
from scipy.optimize import minimize

# === 基本設定 ===
SAMPLERATE = 44100
BUFFER_SIZE = 2048
CHANNEL = 0
audio_buffer = np.zeros(BUFFER_SIZE)

# === EGG信号受信コールバック ===
def audio_callback(indata, frames, time, status):
    global audio_buffer
    audio_buffer = indata[:, CHANNEL]

# === LFモデル定義 ===
def lf_model(t, E1, a, w, E2, b, Te, Tc):
    u = np.zeros_like(t)
    for i, ti in enumerate(t):
        if ti < Te:
            u[i] = E1 * np.exp(a * ti) * np.sin(w * ti)
        elif Te <= ti < Tc:
            u[i] = -E2 * (np.exp(-b * (ti - Te)) - np.exp(-b * (Tc - Te)))
    return u

# === 最小二乗損失関数 ===
def lf_loss(params, t, y):
    E1, a, w, E2, b, Te, Tc = params
    y_pred = lf_model(t, E1, a, w, E2, b, Te, Tc)
    return np.sum((y - y_pred) ** 2)

# === 1周期のフィッティング処理 ===
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

# === 複数周期に適用して平均をとる ===
def extract_parameters(buffer, fs):
    deg = np.diff(buffer)
    peaks, _ = find_peaks(-deg, distance=int(fs / 880))
    oqs, rqs, alphas = [], [], []
    for i in range(len(peaks) - 1):
        start, end = peaks[i], peaks[i+1]
        cycle = deg[start:end]
        if len(cycle) < 100:
            continue
        oq, rq, alpha = fit_lf_model(cycle, fs)
        if oq and 0 < oq < 1 and 0 < rq < 1 and 0 < alpha < 2:
            oqs.append(oq)
            rqs.append(rq)
            alphas.append(alpha)
    return oqs, rqs, alphas

# === 描画設定 ===
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))
x = np.arange(BUFFER_SIZE)
line1, = ax1.plot(x, np.zeros(BUFFER_SIZE))
ax1.set_ylim(-1.0, 1.0)
ax1.set_title("Real-time EGG signal")
ax1.set_ylabel("Amplitude")

bar = ax2.bar(["Oq", "Rq", "αm"], [0.5, 0.01, 0.7])
ax2.set_ylim(0, 1)
ax2.set_ylabel("Value")
ax2.set_title("Estimated Glottal Parameters (ARX-LF)")

# === アニメーション関数 ===
def update_plot(frame):
    line1.set_ydata(audio_buffer)
    oqs, rqs, alphas = extract_parameters(audio_buffer, SAMPLERATE)
    if oqs:
        bar[0].set_height(np.mean(oqs))
        bar[1].set_height(np.mean(rqs))
        bar[2].set_height(np.mean(alphas))
    return line1, *bar

# === ストリーム起動と表示 ===
stream = sd.InputStream(callback=audio_callback, channels=2, samplerate=SAMPLERATE, blocksize=BUFFER_SIZE)
with stream:
    ani = FuncAnimation(fig, update_plot, interval=300)
    plt.tight_layout()
    plt.show()

