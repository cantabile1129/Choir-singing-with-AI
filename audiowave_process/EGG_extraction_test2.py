import numpy as np
import sounddevice as sd
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import minimize
import time

# === 基本設定 ===
SAMPLERATE = 44100
BUFFER_SIZE = 2048
CHANNEL = 0  # モノラルで十分
audio_buffer = np.zeros(BUFFER_SIZE)

# === 音声デバイス表示（起動時に一度だけ） ===
print("🔊 使用可能なオーディオデバイス一覧:")
print(sd.query_devices(), flush=True)

# === コールバック関数 ===
def audio_callback(indata, frames, time_info, status):
    global audio_buffer
    audio_buffer = indata[:, CHANNEL]
    print(f"🟢 audio_callback called. mean={np.mean(audio_buffer):.4f}", flush=True)

# === LFモデル定義 ===
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

# === フィッティング処理 ===
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
        return {
            "Oq": Oq,
            "Rq": Rq,
            "αm": alpha_m,
            "T0": T0,
            "Tp": Tp,
            "Te": Te,
            "Tc": Tc
        }
    else:
        print("⚠️ 最適化に失敗しました。", flush=True)
        return None

# === パラメータ抽出処理 ===
def extract_parameters(buffer, fs):
    print("🛠 extract_parameters 実行", flush=True)
    deg = gaussian_filter1d(np.diff(buffer), sigma=2)
    peaks, _ = find_peaks(-deg, distance=int(fs / 800), prominence=0.01)
    print(f"🔍 検出されたピーク数: {len(peaks)}", flush=True)
    for i in range(len(peaks) - 1):
        start, end = peaks[i], peaks[i + 1]
        cycle = deg[start:end:2]  # 間引き
        if len(cycle) < 100:
            continue
        params = fit_lf_model(cycle, fs)
        if params and 0 < params["Oq"] < 1 and 0 < params["Rq"] < 1 and 0 < params["αm"] < 2:
            print(f"✅ [周期 {i+1}] Oq: {params['Oq']:.3f}, Rq: {params['Rq']:.3f}, αm: {params['αm']:.3f}, "
                  f"T0: {params['T0']:.4f}, Tp: {params['Tp']:.4f}, Te: {params['Te']:.4f}, Tc: {params['Tc']:.4f}", flush=True)
            break

# === メイン実行 ===
print("🚀 リアルタイム EGG パラメータ解析（ARX-LFモデル）開始...", flush=True)

stream = sd.InputStream(callback=audio_callback, channels=1, samplerate=SAMPLERATE, blocksize=BUFFER_SIZE)

with stream:
    try:
        while True:
            extract_parameters(audio_buffer.copy(), SAMPLERATE)
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("🛑 終了しました。", flush=True)


