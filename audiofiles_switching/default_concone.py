#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
default_concone.py (ASIO 対応版)

- メトロノーム → 伴奏(+AI L/R) を再生
- 入力は 4ch (mic0, mic1, mic2, egg3) を想定して録音
- 終了後に
    - full_input_ch0.wav ~ full_input_ch3.wav
    - full_record.csv (time, mic, egg)
    - full_record.parquet (なければ feather にフォールバック)
  を保存する

ASIO デバイスは
  --input_device  <ASIO 入力デバイス index>
  --output_device <ASIO 出力デバイス index>
で明示指定する。
"""

import os
import sys
import time
import argparse

import numpy as np
import sounddevice as sd
import soundfile as sf
import pandas as pd

# ==============================
# 設定
# ==============================
SR = 44100
BLOCKSIZE_DEFAULT = 256

METRONOME_WAV = "audiofiles_switching/audiofiles/コンコーネ50番前出しスネアドラム.wav"
ACCOMP_WAV    = "audiofiles_switching/audiofiles/Concone_ピアノ伴奏.wav"

# L/R それぞれの AI 音声
AI_LEFT_WAV   = "audiofiles_switching/audiofiles/default/concone_synthesizerV_JinAI_default_MixDown.wav"
AI_RIGHT_WAV  = "audiofiles_switching/audiofiles/default/concone_synthesizerV_KevinAI_default_MixDown.wav"


def load_wav(path: str, sr: int) -> np.ndarray:
    """ステレオ/モノラル問わず 2D float32 にして返す"""
    if not os.path.exists(path):
        print(f"[WARN] wav not found: {path}", file=sys.stderr)
        return np.zeros((0, 1), dtype=np.float32)

    y, fs = sf.read(path, always_2d=True)
    if fs != sr:
        print(f"[WARN] resampling not implemented, but fs={fs} != {sr}", file=sys.stderr)
        # 必要ならここで librosa.resample 等に差し替え
    return y.astype(np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--outdir",
        type=str,
        default="audiofiles_switching/audio",
        help="録音データの出力先ディレクトリ",
    )
    parser.add_argument(
        "--input_device",
        type=int,
        required=True,
        help="録音に使うデバイス index（ASIO デバイス番号）",
    )
    parser.add_argument(
        "--output_device",
        type=int,
        required=True,
        help="再生に使うデバイス index（ASIO デバイス番号）",
    )
    parser.add_argument(
        "--blocksize",
        type=int,
        default=BLOCKSIZE_DEFAULT,
        help="Stream の blocksize（既定: 256）",
    )
    args = parser.parse_args()

    outdir = args.outdir
    os.makedirs(outdir, exist_ok=True)

    # ==============================
    # wav を全部先読み
    # ==============================
    print("[LOAD] metronome / accompaniment / AI (L/R) をロード中...")
    met_audio = load_wav(METRONOME_WAV, SR)    # shape = (N_met, 2) を想定
    acc_audio = load_wav(ACCOMP_WAV, SR)       # shape = (N_acc, 2)

    ai_left_audio  = load_wav(AI_LEFT_WAV, SR)   # shape = (N_aiL, 1 or 2)
    ai_right_audio = load_wav(AI_RIGHT_WAV, SR)  # shape = (N_aiR, 1 or 2)

    if met_audio.shape[1] < 2:
        met_audio = np.repeat(met_audio, 2, axis=1)
    if acc_audio.shape[1] < 2:
        acc_audio = np.repeat(acc_audio, 2, axis=1)
    if ai_left_audio.shape[1] > 1:
        ai_left_audio = ai_left_audio[:, :1]
    if ai_right_audio.shape[1] > 1:
        ai_right_audio = ai_right_audio[:, :1]

    met_len = met_audio.shape[0]
    acc_len = acc_audio.shape[0]

    # metronome -> accompaniment(+AI) で再生時間を決定
    total_play_samples = met_len + acc_len
    total_play_time = total_play_samples / SR
    print(f"[INFO] total play time ≒ {total_play_time:.3f} sec")

    # ==============================
    # 入力バッファ（4ch: mic0, mic1, mic2, egg3）
    # ==============================
    IN_CH = 4  # ch0-2: mic, ch3: EGG を想定
    full_buffer = np.zeros((total_play_samples, IN_CH), dtype=np.float32)

    # 録音位置（サンプル）
    sample_index = 0
    last_debug_time = 0.0  # DEBUG-REC 表示用

    # ==============================
    # sd.Stream コールバック
    # ==============================
    MIC0_IDX = 0
    EGG_IDX = 3

    def audio_callback(indata, outdata, frames, time_info, status):
        nonlocal sample_index, last_debug_time

        if status:
            print("[SD-STATUS]", status, file=sys.stderr)

        start = sample_index
        end = start + frames

        # ---- DEBUG-REC: 0.5 秒ごとに入力レベル表示 ----
        t_now = start / SR
        if t_now - last_debug_time >= 0.5:
            if indata.size > 0:
                rms = np.sqrt((indata.astype(np.float64) ** 2).mean(axis=0) + 1e-12)
                print(f"[DEBUG-REC] t={t_now:.3f}s | ch_levels={rms.tolist()}")
                print(
                    f"[DEBUG-REC] indata: min={indata.min():.3f}, max={indata.max():.3f}"
                )
            last_debug_time = t_now

        # ---- 録音（full_buffer に保存）----
        if start < full_buffer.shape[0]:
            write_end = min(end, full_buffer.shape[0])
            frames_to_copy = write_end - start
            if frames_to_copy > 0:
                # indata.shape[1] が IN_CH 未満の時はその分だけコピーされる
                ch_to_copy = min(IN_CH, indata.shape[1])
                full_buffer[start:write_end, :ch_to_copy] = indata[:frames_to_copy, :ch_to_copy]

        # ---- 再生用バッファ初期化 ----
        outdata[:] = 0.0

        # 再生位置が範囲外なら停止
        if start >= total_play_samples:
            raise sd.CallbackStop()

        play_end = min(end, total_play_samples)
        n_frames = play_end - start
        if n_frames <= 0:
            return

        # 出力は常に 2ch (L/R)
        # ここで metronome → accompaniment(+AI) を合成
        for i in range(n_frames):
            pos = start + i
            left = 0.0
            right = 0.0

            if pos < met_len:
                # メトロノームのみ
                left = met_audio[pos, 0]
                right = met_audio[pos, 1]
            elif pos < met_len + acc_len:
                # 伴奏 + AI(L/R)
                acc_pos = pos - met_len
                left = acc_audio[acc_pos, 0]
                right = acc_audio[acc_pos, 1]

                # AI を伴奏と同時に再生（長さが短ければその分だけ）
                ai_pos = acc_pos
                if ai_pos < ai_left_audio.shape[0]:
                    left += float(ai_left_audio[ai_pos, 0])
                if ai_pos < ai_right_audio.shape[0]:
                    right += float(ai_right_audio[ai_pos, 0])

            outdata[i, 0] = left
            outdata[i, 1] = right

        sample_index += frames

    # ==============================
    # デバイス情報の表示（ASIO 確認用）
    # ==============================
    try:
        print(sd.query_hostapis())
        for i, d in enumerate(sd.query_devices()):
            print(i, d["name"], "| hostapi =", sd.query_hostapis()[d["hostapi"]]["name"])

        dev_in = sd.query_devices(args.input_device)
        dev_out = sd.query_devices(args.output_device)
        print(
            f"[INFO] ASIO input_device={args.input_device} "
            f"(max_in={dev_in['max_input_channels']})"
        )
        print(
            f"[INFO] ASIO output_device={args.output_device} "
            f"(max_out={dev_out['max_output_channels']})"
        )
    except Exception as e:
        print("[WARN] failed to query devices/hostapis:", e, file=sys.stderr)

    # ==============================
    # ストリームオープン & 実行 (ASIO)
    # ==============================
    stream_kwargs = dict(
        samplerate=SR,
        blocksize=args.blocksize,
        channels=(IN_CH, 2),
        dtype="float32",
        callback=audio_callback,
        device=(args.input_device, args.output_device),  # ★ ASIO: 入力/出力を分けて指定
    )

    print("[STREAM] start (ASIO)")
    try:
        with sd.Stream(**stream_kwargs):
            # 再生が終わるまで待機（少し余裕を持たせる）
            while sample_index < total_play_samples:
                time.sleep(0.1)
    except Exception as e:
        print("[ERROR] Failed to start ASIO stream:", e, file=sys.stderr)
        return
    print("[STREAM] finished")

    # ==============================
    # チャンネルごとの wav 保存
    # ==============================
    for ch in range(IN_CH):
        ch_wav = os.path.join(outdir, f"full_input_ch{ch}.wav")
        sf.write(ch_wav, full_buffer[:, ch], SR)
        print(f"[SAVE] {ch_wav}")

    # ==============================
    # CSV (time, mic0, egg3)
    # ==============================
    times = np.arange(full_buffer.shape[0], dtype=np.float64) / SR
    mic0 = full_buffer[:, MIC0_IDX]
    egg = full_buffer[:, EGG_IDX] if IN_CH > EGG_IDX else np.zeros_like(mic0)

    df = pd.DataFrame({"time": times, "mic": mic0, "egg": egg})
    csv_path = os.path.join(outdir, "full_record.csv")
    df.to_csv(csv_path, index=False)
    print(f"[SAVE] {csv_path}")

    # ==============================
    # Parquet（pyarrow / fastparquet がない環境でも保存できる版）
    # ==============================
    pq_path = os.path.join(outdir, "full_record.parquet")

    try:
        # pyarrow があればそのまま使う
        df.to_parquet(pq_path, engine="pyarrow")
        print(f"[SAVE] {pq_path}  (engine=pyarrow)")
    except Exception:
        try:
            # fastparquet があれば fallback
            df.to_parquet(pq_path, engine="fastparquet")
            print(f"[SAVE] {pq_path}  (engine=fastparquet)")
        except Exception:
            # どちらも無い場合 → parquet 互換の feather で代わりに保存
            feather_path = pq_path.replace(".parquet", ".feather")
            try:
                df.to_feather(feather_path)
                print(f"[SAVE] {feather_path}  (fallback feather)")
            except Exception as e:
                print(f"[ERROR] unable to save parquet/feather: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()