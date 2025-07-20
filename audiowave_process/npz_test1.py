import numpy as np


data = np.load("rawdata/recorded_sync.npz")
mic = data["mic_signal"]
egg = data["egg_signal"]
time = data["time_sec"]
