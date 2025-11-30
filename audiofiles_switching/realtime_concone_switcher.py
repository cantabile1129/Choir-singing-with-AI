#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
realtime_concone_switcher_segment_range.py
ユーザがコンコーネを歌い、指定された時間区間ごとに録音し、
real_file_selector.py を用いて次の AI 音声ファイルを決定し、
Clarett 8Pre 経由で再生する。

- メトロノーム時間(最初の6秒など)は録音しない（--segments に 0-x を指定しなければOK）
- 区間は --segments "6-12" "12-18" ... のように指定
- ログと録音データは --outdir 以下に保存
- AI 音声ファイルは --ai_root から検索
"""

import argparse
import os
import sys
import csv
import tempfile
from typing import List, Tuple

import numpy as np
import sounddevice as sd
import soundfile as sf

import real_file_selector as rfs


# --------------------------------------------
# 録音 / WAV 生成
# --------------------------------------------

def record_exact_time(start_sec: float, stop_sec: float,
                      sr: int, channels: int, device: int | None) -> np.ndarray:
    """
    start_sec〜stop_sec の区間だけ録音する。
    """
    duration = stop_sec - start_sec
    print(f"[REC] {duration:.3f} 秒録音 (区間 {start_sec}-{stop_sec} sec)")

    if device is not None:
        sd.default.device = (device, device)

    rec = sd.rec(
        int(duration * sr),
        samplerate=sr,
        channels=channels,
        dtype="float32",
    )
    sd.wait()
    print("[REC] 録音終了")
    return rec


def write_temp_wav(y: np.ndarray, sr: int, outdir: str, seg_index: int) -> str:
    """
    区間録音を outdir に保存して実ファイルパスを返す
    """
    if y.ndim > 1:
        y_mono = y.mean(axis=1)
    else:
        y_mono = y

    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, f"record_seg{seg_index:02d}.wav")
    sf.write(path, y_mono, sr)
    return path


# --------------------------------------------
# AI音声ファイル決定
# --------------------------------------------

def select_ai_file(
    wav_path: str,
    reg_bank: rfs.RegressionBank,
    perc_bank: rfs.PercentileBank,
    gmm_bank: rfs.GMMParams,
    oq_cand: List[int],
    rq_cand: List[int],
    base_prefix: str,
    next_seg: int,
    ai_root: str,
) -> Tuple[int, int, int, str, dict]:

    feat = rfs.extract_acoustic_features(wav_path)

    oq_idx, rq_idx, debug_oqrq = rfs.select_oq_rq_indices(
        features=feat,
        reg_bank=reg_bank,
        perc_bank=perc_bank,
        gmm_bank=gmm_bank,
        oq_candidates=oq_cand,
        rq_candidates=rq_cand,
    )

    vib_idx, debug_vib = rfs.select_vibrato_index_from_percentiles(
        vibrato_cents_abs=feat["vibrato_cents_abs"],
        perc_bank=perc_bank,
    )

    fname = rfs.build_audio_filename(
        base_prefix=base_prefix,
        oq_idx=oq_idx,
        rq_idx=rq_idx,
        vib_idx=vib_idx,
        segment_index=next_seg,
        ext=".wav",
    )

    # AIルートから検索
    ai_path = os.path.join(ai_root, fname)

    debug = {
        "features": feat,
        "oqrq": debug_oqrq,
        "vib": debug_vib,
    }

    return oq_idx, rq_idx, vib_idx, ai_path, debug


# --------------------------------------------
# メイン処理
# --------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Real-time Concone switcher with segment ranges")

    # ===== モデル =====
    parser.add_argument("--percentiles_csv", type=str, required=True)
    parser.add_argument("--extended_csv",    type=str, required=True)
    parser.add_argument("--reg_spline_csv",  type=str, required=True)
    parser.add_argument("--gmm_csv",         type=str, required=True)

    parser.add_argument("--segment_id", type=str, default="segALL")
    parser.add_argument("--quality",    type=str, default="CLEAN")

    # ===== パラメータ候補 =====
    parser.add_argument("--oq_candidates", type=int, nargs="+", default=[1,2,3,4,5])
    parser.add_argument("--rq_candidates", type=int, nargs="+", default=[1,2,3,4,5])

    # ===== セグメント範囲 =====
    parser.add_argument(
        "--segments", nargs="+", required=True,
        help='例: --segments "6-12" "12-18" "18-24"'
    )

    # ===== AI音声ファイルの検索根 =====
    parser.add_argument("--ai_root", type=str, required=True)

    # ===== ベース名 =====
    parser.add_argument("--base_prefix", type=str, required=True)

    # ===== 録音・再生 =====
    parser.add_argument("--sr", type=int, default=48000)
    parser.add_argument("--in_channels", type=int, default=1)
    parser.add_argument("--device", type=int, default=None)

    # ===== ログ =====
    parser.add_argument("--outdir", type=str, required=True)

    args = parser.parse_args()

    # ===== 構造準備 =====
    os.makedirs(args.outdir, exist_ok=True)
    log_csv = os.path.join(args.outdir, "realtime_switch_log.csv")

    # ===== モデル読み込み =====
    reg_bank = rfs.RegressionBank(args.reg_spline_csv)
    perc_bank = rfs.PercentileBank(
        percentiles_csv=args.percentiles_csv,
        extended_csv=args.extended_csv,
        segment_id=args.segment_id,
        quality=args.quality,
    )
    gmm_bank = rfs.load_gmm_params(args.gmm_csv)

    # ===== セグメント解析 =====
    seg_ranges = []
    for s in args.segments:
        a, b = s.split("-")
        seg_ranges.append((float(a), float(b)))

    # ===== ログ初期化 =====
    with open(log_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["seg", "start", "end", "record_path",
                         "oq_idx", "rq_idx", "vib_idx", "ai_file"])

    # ===== メインループ =====
    sr = args.sr

    for i, (st, ed) in enumerate(seg_ranges, start=1):
        print("="*40)
        print(f"[SEG {i}] {st}-{ed} sec")

        # 録音
        rec = record_exact_time(st, ed, sr, args.in_channels, args.device)
        rec_wav = write_temp_wav(rec, sr, args.outdir, i)

        # 次セグメント
        next_seg = i + 1 if i < len(seg_ranges) else i

        if i == len(seg_ranges):
            # 最終セグメントは推定しない
            oq = rq = vib = -1
            ai_path = ""
        else:
            oq, rq, vib, ai_path, debug = select_ai_file(
                wav_path=rec_wav,
                reg_bank=reg_bank,
                perc_bank=perc_bank,
                gmm_bank=gmm_bank,
                oq_cand=args.oq_candidates,
                rq_cand=args.rq_candidates,
                base_prefix=args.base_prefix,
                next_seg=next_seg,
                ai_root=args.ai_root,
            )

            print(f"[SELECT] Next seg{next_seg:02d} -> Oq={oq}, Rq={rq}, vib={vib}")
            print(f"[FILE] {ai_path}")

            # 再生
            if os.path.exists(ai_path):
                audio, fs_file = sf.read(ai_path, always_2d=True)
                if args.device is not None:
                    sd.default.device = (args.device, args.device)
                sd.play(audio, samplerate=sr)
                sd.wait()
            else:
                print(f"[WARN] AI file missing: {ai_path}")

        # ログ追記
        with open(log_csv, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([i, st, ed, rec_wav, oq, rq, vib, ai_path])

    print("[DONE] 全セグメント処理完了")


if __name__ == "__main__":
    main()
