# filesandprocess.py

import subprocess
import os
import time

# === スクリプトパス ===
egg_script = os.path.join("audiowave_process", "rawdata_logger_test1.py")
audio_script = os.path.join("audiofiles_switching", "multichannel_audio4.py")

# === 並列実行 ===
print("🚀 録音・再生の同時実行を開始します")

# 録音プロセスをバックグラウンドで実行
record_proc = subprocess.Popen(["python", egg_script])
time.sleep(1.0)  # 少し待ってから再生を開始（録音バッファ準備のため）

# 再生プロセスをフォアグラウンドで実行（完了まで待機）
subprocess.run(["python", audio_script])

# 録音終了を待機（録音は DURATION 秒かかる）
record_proc.wait()

print("✅ 同時実行が完了しました")
