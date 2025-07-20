#EGGの生の信号をグラフ化．

import numpy as np
import matplotlib.pyplot as plt
import sounddevice as sd

# === 基本設定 ===
SAMPLE_RATE = 44100
BUFFER_SIZE = 1024
CHANNEL = 0  # 0チャンネル目
audio_buffer = np.zeros(BUFFER_SIZE, dtype=np.float32)

# === 音声入力コールバック ===
def audio_callback(indata, frames, time_info, status):
    global audio_buffer
    audio_buffer = indata[:, CHANNEL]

# === リアルタイム表示関数 ===
def visualize_waveform():
    plt.ion()
    fig, ax = plt.subplots(figsize=(10, 4))
    x = np.arange(BUFFER_SIZE)
    line, = ax.plot(x, np.zeros(BUFFER_SIZE))
    ax.set_ylim(-1.0, 1.0)
    ax.set_title("Real-time EGG Signal (Raw Waveform)")
    ax.set_xlabel("Sample")
    ax.set_ylabel("Amplitude")

    while True:
        line.set_ydata(audio_buffer)
        plt.pause(0.05)

# === メイン処理 ===
if __name__ == "__main__":
    print("🎤 リアルタイムでEGG信号（生波形）を表示します。")
    stream = sd.InputStream(callback=audio_callback, channels=1,
                            samplerate=SAMPLE_RATE, blocksize=BUFFER_SIZE)
    with stream:
        visualize_waveform()


