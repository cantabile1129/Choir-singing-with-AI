import numpy as np
import sounddevice as sd

fs = 44100
t = np.linspace(0, 2, fs * 2)
tone = 0.2 * np.sin(2 * np.pi * 440 * t)  # 440Hzトーン

silent = np.zeros_like(tone)

# ch0=a, ch1=b, ch2=c
signal = np.stack([silent, silent, tone], axis=1).astype(np.float32)

sd.play(signal, samplerate=fs)
sd.wait()
