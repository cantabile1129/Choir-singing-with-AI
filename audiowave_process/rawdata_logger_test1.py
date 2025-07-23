import numpy as np
import sounddevice as sd
import soundfile as sf
import pandas as pd
import os

# === パラメータ設定 ===
SAMPLE_RATE = 44100      # 高精度解析用に変更
CHANNELS = 2             # 0: mic, 1: EGG
DURATION = 10            # 秒
BASE_DIR = "recordings"
WAV_PATH = os.path.join(BASE_DIR, "recorded_sync.wav")
CSV_PATH = os.path.join(BASE_DIR, "recorded_sync.csv")
PARQUET_PATH = os.path.join(BASE_DIR, "recorded_sync.parquet")
NPZ_PATH = os.path.join(BASE_DIR, "recorded_sync.npz")
os.makedirs(BASE_DIR, exist_ok=True)

# === 録音処理 ===
print("🔴 録音開始...")
data = sd.rec(int(DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE,
              channels=CHANNELS, dtype='float32')
sd.wait()
print("✅ 録音完了")

# === WAV 保存 ===
sf.write(WAV_PATH, data, SAMPLE_RATE)
print(f"💾 WAV保存完了: {WAV_PATH}")

# === タイムスタンプ生成 ===
timestamps = np.arange(len(data)) / SAMPLE_RATE

# === DataFrame 作成（CSV & Parquet 用）===
df = pd.DataFrame({
    "time_sec": timestamps,
    "mic_signal": data[:, 0],
    "egg_signal": data[:, 1],
})

# === CSV 保存 ===
df.to_csv(CSV_PATH, index=False)
print(f"📄 CSV保存完了: {CSV_PATH}")

# === Parquet 保存 ===
df.to_parquet(PARQUET_PATH, index=False)
print(f"📦 Parquet保存完了: {PARQUET_PATH}")

# === NumPy 圧縮保存（npz）===
np.savez_compressed(NPZ_PATH, time_sec=timestamps,
                    mic_signal=data[:, 0], egg_signal=data[:, 1])
print(f"🗜 NumPy圧縮保存完了: {NPZ_PATH}")
