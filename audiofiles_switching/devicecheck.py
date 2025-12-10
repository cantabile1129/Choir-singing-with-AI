import sounddevice as sd
import soundcard as sc

print("===== SoundDevice (MME/WASAPI/WDM/ASIO) =====")
print(sd.query_devices())

print("\n===== SoundDevice (ASIO only) =====")
try:
    print(sd.query_devices(kind='asio'))
except Exception as e:
    print("ASIO backend not available:", e)

print("\n===== SoundCard (Speakers) =====")
for sp in sc.all_speakers():
    print(f"{sp.name}  ({sp.channels} ch)")

print("\n===== SoundCard (Microphones) =====")
for mic in sc.all_microphones():
    print(f"{mic.name}  ({mic.channels} ch)")
