import soundfile as sf
import numpy as np
import sounddevice as sd

# === ファイルパス ===
common_path = "audiofiles_switching/audiofiles/コンコーネ50番前出しスネアドラム.wav"
a_path = "audiofiles_switching/audiofiles/concone_synthesizerV_KevinAI_trial2_MixDown.wav"
b_path = "audiofiles_switching/audiofiles/concone_synthesizerV_京町セイカAI_trial2_MixDown.wav"

# === 共通音声を読み込み（6秒分だけ）===
common, sr_c = sf.read(common_path)
common = common[:sr_c * 6]  # 6秒だけに切り取る
if common.ndim > 1:
    common = common.mean(axis=1)  # モノラルに変換（必要であれば）

# === 個別音声を読み込み（モノラル） ===
a, sr1 = sf.read(a_path)
b, sr2 = sf.read(b_path)
assert sr1 == sr2 == sr_c, "サンプリングレートが一致していません"

# === 長さ揃える（共通音声と個別音声）===
length = max(len(a), len(b))
a = np.pad(a, (0, length - len(a)))
b = np.pad(b, (0, length - len(b)))

# === ステレオ信号を作成 ===
common_stereo = np.stack([common, common], axis=1).astype(np.float32)
individual_stereo = np.stack([a, b], axis=1).astype(np.float32)

# === 再生 ===
print("🔊 共通音声 再生開始...")
sd.play(common_stereo, samplerate=sr_c)
sd.wait()

print("🔊 個別音声 再生開始（左右）...")
sd.play(individual_stereo, samplerate=sr_c)
sd.wait()

print("✅ すべての再生が完了しました")
