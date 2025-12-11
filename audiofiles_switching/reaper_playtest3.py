import sounddevice as sd

print(sd.query_hostapis())
for i, d in enumerate(sd.query_devices()):
    print(i, d['name'], d)
