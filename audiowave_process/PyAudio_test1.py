#なんの参考になるかはわからんが，拾ってきたらしい．

import pyaudio
import numpy as np
from pydub import AudioSegment
import io
from pydub.effects import low_pass_filter, high_pass_filter
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.fftpack import fft, ifft
from scipy.signal import resample
import threading

CHUNK = 1024
FORMAT = pyaudio.paFloat32
CHANNELS = 1
RATE = 44100

p = pyaudio.PyAudio()

# グローバル変数
audio_data = np.zeros(CHUNK)
noise_profile = None

def apply_effects(audio):
    # ローパスフィルター
    audio = low_pass_filter(audio, 1000)
    # ハイパスフィルター
    audio = high_pass_filter(audio, 300)
    return audio

def noise_reduction(audio_data):
    global noise_profile
    
    # ノイズプロファイルの更新（最初の数フレーム）
    if noise_profile is None:
        noise_profile = np.abs(fft(audio_data))
        return audio_data
    
    # スペクトル減算
    spectrum = fft(audio_data)
    noise_reduction_factor = 0.9
    reduced_spectrum = spectrum - noise_reduction_factor * noise_profile
    reduced_spectrum = np.maximum(reduced_spectrum, 0)
    
    return np.real(ifft(reduced_spectrum))

def pitch_shift(audio_data, semitones):
    factor = 2 ** (semitones / 12)
    stretched = resample(audio_data, int(len(audio_data) / factor))
    return resample(stretched, len(audio_data))

def process_audio(audio_data):
    # NumPy配列をpydubのAudioSegmentに変換
    audio = AudioSegment(
        audio_data.tobytes(), 
        frame_rate=RATE,
        sample_width=audio_data.dtype.itemsize, 
        channels=CHANNELS
    )
    
    # エフェクトを適用
    processed_audio = apply_effects(audio)
    
    # ノイズリダクション
    processed_audio = noise_reduction(np.array(processed_audio.get_array_of_samples()).astype(np.float32))
    
    # ピッチシフト
    processed_audio = pitch_shift(processed_audio, semitones=2)
    
    return processed_audio

def callback(in_data, frame_count, time_info, status):
    global audio_data
    audio_data = np.frombuffer(in_data, dtype=np.float32)
    processed_data = process_audio(audio_data)
    return (processed_data.astype(np.float32).tobytes(), pyaudio.paContinue)

#音声ストリームの設定
stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                output=True,
                frames_per_buffer=CHUNK,
                stream_callback=callback)

# 可視化の設定
fig, ax = plt.subplots()
x = np.arange(0, CHUNK)
line, = ax.plot(x, np.random.rand(CHUNK))

def update_plot(frame):
    global audio_data
    line.set_ydata(audio_data)
    return line,

# アニメーションの設定（修正後）
ani = FuncAnimation(fig, update_plot, blit=True, cache_frame_data=False)

# 音声処理スレッド
def audio_processing_thread():
    stream.start_stream()
    while stream.is_active():
        pass

# メイン処理
if __name__ == "__main__":
    processing_thread = threading.Thread(target=audio_processing_thread)
    processing_thread.start()

    try:
        plt.show()
    except KeyboardInterrupt:
        stream.stop_stream()
        stream.close()
        p.terminate()
        processing_thread.join()
