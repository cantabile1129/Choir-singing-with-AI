import numpy as np
import sounddevice as sd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# --- 設定 ---
SAMPLERATE = 44100  # またはデバイスに合わせて変更
BUFFER_SIZE = 1024  # 一度に取得するサンプル数
CHANNEL = 0         # モノラル（ステレオ左 = 0、右 = 1）

# グローバルバッファ（更新用）
audio_buffer = np.zeros(BUFFER_SIZE)

# --- 音声入力コールバック ---
def audio_callback(indata, frames, time, status):
    global audio_buffer
    audio_buffer = indata[:, CHANNEL]

# --- プロットの初期化 ---
fig, ax = plt.subplots()
x = np.arange(BUFFER_SIZE)
line, = ax.plot(x, np.zeros(BUFFER_SIZE))
ax.set_ylim(-1.0, 1.0)
ax.set_title("Real-time EGG signal")
ax.set_xlabel("Samples")
ax.set_ylabel("Amplitude")

# --- アニメーション更新関数 ---
def update_plot(frame):
    line.set_ydata(audio_buffer)
    return line,

# --- 音声ストリーム開始と可視化 ---
stream = sd.InputStream(callback=audio_callback, channels=2, samplerate=SAMPLERATE, blocksize=BUFFER_SIZE)
with stream:
    ani = FuncAnimation(fig, update_plot, interval=30)
    plt.show()
