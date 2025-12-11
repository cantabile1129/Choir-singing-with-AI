import sounddevice as sd

rate = 44100
frames = 256

def callback(indata, outdata, frames, time, status):
    if status:
        print("STATUS:", status)
    # 入力をそのまま出力（エコー）
    outdata[:] = indata[:, :2]  # 出力は2chだけ使う

stream = sd.Stream(
    samplerate=rate,
    blocksize=frames,
    dtype="float32",
    channels=(8, 2),  # 入力8ch, 出力2ch
    device=(14, 14)  # ASIO デバイス index
)

with stream:
    print("ASIO stream opened successfully. Press Ctrl+C to stop.")
    sd.sleep(3000)  # 3秒テスト

