import soundfile as sf
import numpy as np
import sounddevice as sd

# === ファイルパス ===
common_path = "audiofiles_switching/audiofiles/コンコーネ50番前出しスネアドラム.wav"
a_path = "audiofiles_switching/audiofiles/concone_synthesizerV_KevinAI_trial2_MixDown.wav"
b_path = "audiofiles_switching/audiofiles/concone_synthesizerV_京町セイカAI_trial2_MixDown.wav"
c_path = "audiofiles_switching/audiofiles/Concone_ピアノ伴奏.wav"

# === 音声読み込み ===
common, sr_common = sf.read(common_path)
a, sr1 = sf.read(a_path)
b, sr2 = sf.read(b_path)
c, sr3 = sf.read(c_path)

assert sr1 == sr2 == sr3 == sr_common, "サンプリングレートが一致していません"
fs = sr_common

# === 共通音声（6秒） + モノラル化 ===
common = common[:fs * 6]
if common.ndim > 1:
    common = common.mean(axis=1)

# === 個別音声の長さを揃える ===
max_len = max(len(a), len(b), len(c))

def pad_to_length(x, length):
    return np.pad(x if x.ndim == 1 else x.mean(axis=1), (0, length - len(x)))

a = pad_to_length(a, max_len)
b = pad_to_length(b, max_len)
c = pad_to_length(c, max_len)

# === 共通音声を3chへ複製 ===
common_3ch = np.stack([common, common, common], axis=1).astype(np.float32)

# === 個別3ch音声を構成 ===
individual_3ch = np.stack([a, b, c], axis=1).astype(np.float32)

# === 時間方向に連結（6秒共通 → 個別）===
combined = np.concatenate([common_3ch, individual_3ch], axis=0)

# === 再生 ===
print("🔊 3チャンネル完全同期再生を開始...")
sd.play(combined, samplerate=fs)
sd.wait()
print("✅ すべての再生が完了しました（3ch）")


