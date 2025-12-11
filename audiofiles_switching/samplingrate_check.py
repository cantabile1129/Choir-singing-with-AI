import soundfile as sf
import sys

path = "audiofiles_switching/audiofiles/Saki/concone_synthesizerV_SakiAI_1_1_1_seg03.wav"
y, fs = sf.read(path)
print(fs)
