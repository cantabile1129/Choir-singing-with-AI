#!/usr/bin/env python3
import argparse
from pathlib import Path
import soundfile as sf
import numpy as np
import glob
import re


# 数字3つを抽出する正規表現（例：_1_2_3 の部分）
TRIPLE_NUMBER_PATTERN = re.compile(r".*_(\d+)_(\d+)_(\d+)$")


def parse_synthv_threeparams(stem):
    """
    ファイル名から prefix, A, B, vib を抽出（数字3つのみ対応）。
    例：
    concone_synthesizerV_小春AI_1_1_1 → prefix=concone_synthesizerV_小春AI
                                         A=1, B=1, vib=1
    """
    m = TRIPLE_NUMBER_PATTERN.match(stem)
    if not m:
        raise ValueError(f"File does not match *_A_B_VIB pattern: {stem}")

    A = int(m.group(1))
    B = int(m.group(2))
    vib = int(m.group(3))

    prefix = stem[: stem.rfind(f"_{A}_{B}_{vib}")]

    return prefix, A, B, vib


def split_audio(audio, sr, split_points):
    segments = []
    n_samples = len(audio)
    pts_samples = [int(t * sr) for t in split_points]

    for i in range(len(pts_samples) - 1):
        start = pts_samples[i]
        end = pts_samples[i + 1]
        segments.append(audio[start:end])

    last_start = pts_samples[-1]
    segments.append(audio[last_start:n_samples])
    return segments


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--audio",
        type=str,
        required=True,
        help="glob pattern (例：path/*.wav)"
    )
    parser.add_argument(
        "--splits",
        type=float,
        nargs="+",
        required=True,
        help="split points in seconds"
    )
    parser.add_argument(
        "--outdir",
        type=str,
        required=True
    )
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # glob パターン取得
    candidates = sorted(glob.glob(args.audio))
    if not candidates:
        raise FileNotFoundError(f"No files matched: {args.audio}")

    # 数字三つルールに合うものだけフィルタ
    file_list = []
    for f in candidates:
        stem = Path(f).stem
        if TRIPLE_NUMBER_PATTERN.match(stem):
            file_list.append(f)

    if not file_list:
        raise FileNotFoundError("No files with three-number pattern A_B_VIB found.")

    print(f"[INFO] Found {len(file_list)} valid files.")

    # 分割処理
    for file in file_list:
        path = Path(file)
        audio, sr = sf.read(path)
        prefix, A, B, vib = parse_synthv_threeparams(path.stem)

        print(f"[PROCESS] {path.name} → A={A}, B={B}, vib={vib}")

        segments = split_audio(audio, sr, args.splits)

        for i, seg in enumerate(segments):
            segid = f"seg{i+1:02d}"
            new_name = f"{prefix}_{A}_{B}_{vib}_{segid}.wav"
            out_path = outdir / new_name
            sf.write(out_path, seg, sr)
            print(f"     saved: {new_name}")

    print("\n[DONE]")


if __name__ == "__main__":
    main()
