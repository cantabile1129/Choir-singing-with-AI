import numpy as np
import matplotlib.pyplot as plt
import sounddevice as sd
import librosa
import librosa.display
import scipy.signal
import queue
import threading
import time
import soundfile as sf
import os

# === パラメータ設定 ===
SAMPLE_RATE = 16000
CHANNELS = 1
#実際のコンコーネは78秒（24小節＋前出し2小節＋BPM80）
BUFFER_DURATION = 5  # 秒
CHUNK_SIZE = 1024
BUFFER_SIZE = SAMPLE_RATE * BUFFER_DURATION
audio_buffer = np.zeros(BUFFER_SIZE, dtype=np.float32)
audio_queue = queue.Queue()
save_path = "recorded.wav"
feature_dir = "features"
os.makedirs(feature_dir, exist_ok=True)

# === 音声コールバック ===
def audio_callback(indata, frames, time_info, status):
    global audio_buffer
    audio_queue.put(indata.copy())
    audio_buffer = np.roll(audio_buffer, -frames)
    audio_buffer[-frames:] = indata[:, 0]

# === フォルマント抽出 ===
def extract_formants(signal, sr):
    try:
        a = librosa.lpc(signal, order=12)
        w, h = scipy.signal.freqz(1, a, worN=512, fs=sr)
        angles = np.unwrap(np.angle(h))
        formants = w[np.where(np.abs(np.diff(angles)) > 1.0)]
    except Exception:
        formants = np.array([])
    return formants

# === 特徴量抽出 ===
def extract_features(signal, sr):
    mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=13)
    f0, _, _ = librosa.pyin(signal, fmin=50, fmax=500)
    spec = librosa.stft(signal)
    spec_db = librosa.amplitude_to_db(np.abs(spec), ref=np.max)
    spectral_envelope = np.mean(np.abs(spec), axis=1)

    bands = [(0, 500), (500, 1000), (1000, 2000), (2000, 4000), (4000, 8000)]
    band_energy = []
    freqs = librosa.fft_frequencies(sr=sr)
    mag = np.abs(spec)
    for low, high in bands:
        idx = np.where((freqs >= low) & (freqs < high))[0]
        energy = np.sum(mag[idx, :] ** 2)
        band_energy.append(energy)

    formants = extract_formants(signal, sr)
    power = np.mean(signal ** 2)

    return mfcc, f0, spec_db, spectral_envelope, band_energy, formants, power

# === リアルタイム可視化スレッド ===
def visualize():
    plt.ion()
    fig, axs = plt.subplots(6, 1, figsize=(12, 18))

    while True:
        signal = audio_buffer.copy()
        mfcc, f0, spec_db, envelope, band_energy, formants, power = extract_features(signal, SAMPLE_RATE)

        axs[0].cla()
        librosa.display.specshow(mfcc, sr=SAMPLE_RATE, x_axis='time', ax=axs[0])
        axs[0].set_title("MFCC")

        axs[1].cla()
        axs[1].imshow(spec_db, aspect='auto', origin='lower')
        axs[1].set_title("Spectrogram (dB)")

        axs[2].cla()
        axs[2].plot(envelope)
        axs[2].set_title("Spectral Envelope")

        axs[3].cla()
        axs[3].bar([f"{low}-{high}Hz" for (low, high) in [(0,500),(500,1000),(1000,2000),(2000,4000),(4000,8000)]], band_energy)
        axs[3].set_title(f"Band Energy (Power={power:.5f})")

        axs[4].cla()
        if f0 is not None:
            axs[4].plot(f0)
        axs[4].set_title("Fundamental Frequency (F0)")

        axs[5].cla()
        if formants is not None and len(formants) > 0:
            axs[5].stem(formants, np.ones_like(formants))
        axs[5].set_title("Formants (Estimated)")

        plt.tight_layout()
        plt.pause(0.1)

# === 音声録音（保存）===
def record_audio():
    recorded = []
    start = time.time()
    while time.time() - start < BUFFER_DURATION:
        try:
            data = audio_queue.get(timeout=1)
            recorded.append(data)
        except queue.Empty:
            pass
    full_audio = np.concatenate(recorded, axis=0)
    sf.write(save_path, full_audio, SAMPLE_RATE)
    print(f"✔ 録音を保存しました: {save_path}")
    return full_audio

# === メイン処理 ===
if __name__ == "__main__":
    stream = sd.InputStream(callback=audio_callback, channels=CHANNELS,
                            samplerate=SAMPLE_RATE, blocksize=CHUNK_SIZE)
    stream.start()

    visual_thread = threading.Thread(target=visualize, daemon=True)
    visual_thread.start()

    recorded_data = record_audio()

    # 特徴量を保存（個別）
    mfcc, f0, spec_db, envelope, band_energy, formants, power = extract_features(recorded_data, SAMPLE_RATE)
    np.save(os.path.join(feature_dir, "mfcc.npy"), mfcc)
    np.save(os.path.join(feature_dir, "f0.npy"), f0)
    np.save(os.path.join(feature_dir, "spectrogram_db.npy"), spec_db)
    np.save(os.path.join(feature_dir, "spectral_envelope.npy"), envelope)
    np.save(os.path.join(feature_dir, "band_energy.npy"), band_energy)
    np.save(os.path.join(feature_dir, "formants.npy"), formants)
    np.save(os.path.join(feature_dir, "power.npy"), np.array([power]))

    stream.stop()
    stream.close()
    print("✔ 特徴量を個別に保存しました（features/ ディレクトリ）")
