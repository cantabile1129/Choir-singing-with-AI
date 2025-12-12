#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import time
import argparse

import numpy as np
import sounddevice as sd
import soundfile as sf
import pandas as pd

SR = 44100
BLOCKSIZE = 256

METRONOME_WAV = "audiofiles_switching/audiofiles/コンコーネ50番前出しスネアドラム.wav"
ACCOMP_WAV    = "audiofiles_switching/audiofiles/Concone_ピアノ伴奏.wav"


def load_wav(path: str, sr: int) -> np.ndarray:
    if not os.path.exists(path):
        print(f"[WARN] wav not found: {path}", file=sys.stderr)
        return np.zeros((0, 1), dtype=np.float32)
    y, fs = sf.read(path, always_2d=True)
    if fs != sr:
        print(f"[WARN] resampling not implemented, but fs={fs} != {sr}", file=sys.stderr)
    return y.astype(np.float32)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--outdir", type=str,
        default="audiofiles_switching/audio",
        help="録音データの出力先ディレクトリ")

    # ★ ASIO input / output device
    parser.add_argument("--input_device",  type=int, required=True)
    parser.add_argument("--output_device", type=int, required=True)

    args = parser.parse_args()

    outdir = args.outdir
    os.makedirs(outdir, exist_ok=True)

    print("[LOAD] metronome / accompaniment をロード中...")
    met_audio = load_wav(METRONOME_WAV, SR)
    acc_audio = load_wav(ACCOMP_WAV, SR)

    if met_audio.shape[1] < 2:
        met_audio = np.repeat(met_audio, 2, axis=1)
    if acc_audio.shape[1] < 2:
        acc_audio = np.repeat(acc_audio, 2, axis=1)

    met_len = met_audio.shape[0]
    acc_len = acc_audio.shape[0]
    total_play_samples = met_len + acc_len
    total_play_time = total_play_samples / SR
    print(f"[INFO] total play time ≒ {total_play_time:.3f} sec")

    IN_CH = 4
    MIC0_IDX = 0
    EGG_IDX  = 3

    full_buffer = np.zeros((total_play_samples, IN_CH), dtype=np.float32)

    sample_index = 0
    last_debug_time = 0.0

    def audio_callback(indata, outdata, frames, time_info, status):
        nonlocal sample_index, last_debug_time

        if status:
            print("[SD-STATUS]", status, file=sys.stderr)

        start = sample_index
        end   = start + frames

        # ---- DEBUG ----
        t_now = start / SR
        if t_now - last_debug_time >= 0.5:
            if indata.size > 0:
                rms = np.sqrt((indata.astype(np.float64) ** 2).mean(axis=0) + 1e-12)
                print(f"[DEBUG-REC] t={t_now:.3f}s | ch_levels={rms.tolist()}")
                print(f"[DEBUG-REC] indata: min={indata.min():.3f}, max={indata.max():.3f}")
            last_debug_time = t_now

        # ---- 録音 ----
        if start < full_buffer.shape[0]:
            write_end = min(end, full_buffer.shape[0])
            frames_to_copy = write_end - start
            if frames_to_copy > 0:
                full_buffer[start:write_end, :] = indata[:frames_to_copy, :IN_CH]

        outdata[:] = 0.0

        if start >= total_play_samples:
            raise sd.CallbackStop()

        play_end = min(end, total_play_samples)
        n_frames = play_end - start
        if n_frames <= 0:
            return

        # ---- 再生 ----
        for i in range(n_frames):
            pos = start + i
            if pos < met_len:
                outdata[i] = met_audio[pos]
            else:
                acc_pos = pos - met_len
                outdata[i] = acc_audio[acc_pos]

        sample_index += frames


    # ==============================
    # ASIO stream
    # ==============================
    stream_kwargs = dict(
        samplerate=SR,
        blocksize=BLOCKSIZE,
        channels=(IN_CH, 2),
        dtype="float32",
        callback=audio_callback,
        device=(args.input_device, args.output_device),  # ★ 変更点
    )

    print("[STREAM] start (ASIO)")
    with sd.Stream(**stream_kwargs):
        while sample_index < total_play_samples:
            time.sleep(0.1)
    print("[STREAM] finished")

    # ==============================
    # 保存
    # ==============================
    for ch in range(IN_CH):
        path = os.path.join(outdir, f"full_input_ch{ch}.wav")
        sf.write(path, full_buffer[:, ch], SR)
        print(f"[SAVE] {path}")

    df = pd.DataFrame({
        "time": np.arange(full_buffer.shape[0]) / SR,
        "mic": full_buffer[:, MIC0_IDX],
        "egg": full_buffer[:, EGG_IDX],
    })

    csv_path = os.path.join(outdir, "full_record.csv")
    df.to_csv(csv_path, index=False)
    print(f"[SAVE] {csv_path}")

    pq_path = os.path.join(outdir, "full_record.parquet")
    try:
        df.to_parquet(pq_path)
        print(f"[SAVE] {pq_path}")
    except Exception as e:
        print(f"[WARN] to_parquet failed: {e}")


if __name__ == "__main__":
    main()
