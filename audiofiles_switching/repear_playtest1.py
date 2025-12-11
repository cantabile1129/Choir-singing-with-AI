import sounddevice as sd
import soundfile as sf

data, fs = sf.read("audiofiles_switching/audiofiles/Concone_ピアノ伴奏.wav", dtype='float32')

# 例：ReaRoute5/6 を指定（ステレオ出力）
sd.play(data, samplerate=fs, device=5)
sd.wait()
