#!/usr/bin/env python3
import argparse
from pathlib import Path
import shutil

def parse_synthv_threeparams(stem):
    """
    ファイル名から prefix, A, B, vib を抽出する。
    例：
        concone_synthesizerV_KevinAI_1_1_3_MixDown
        → prefix="concone_synthesizerV_KevinAI"
          A=1, B=1, vib=3
    """
    parts = stem.split("_")

    # 数字になっている部分を探す
    numeric_indices = []
    for i, p in enumerate(parts):
        try:
            int(p)
            numeric_indices.append(i)
        except:
            pass

    if len(numeric_indices) < 3:
        raise ValueError(f"File name does not contain 3 numeric parts: {stem}")

    # 最後の3つを A, B, vib とみなす
    A_idx, B_idx, vib_idx = numeric_indices[-3:]

    A = int(parts[A_idx])
    B = int(parts[B_idx])
    vib = int(parts[vib_idx])

    # prefix は数字の直前まで
    prefix = "_".join(parts[:A_idx])

    return prefix, A, B, vib


def main():
    parser = argparse.ArgumentParser(
        description="Generate 25 SynthV files by replacing A,B with 1–5 and keeping original vib."
    )
    parser.add_argument(
        "--audio",
        type=str,
        nargs="+",
        required=True,
        help="複数の wav ファイルを入力"
    )
    parser.add_argument(
        "--outdir",
        type=str,
        required=True,
        help="出力先ディレクトリ"
    )
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------
    # vibごとにコピー元ファイルを決める
    # -------------------------------------------
    vib_sources = {}   # vib → Path

    for audio_str in args.audio:
        p = Path(audio_str)
        prefix, A, B, vib = parse_synthv_threeparams(p.stem)

        print(f"[READ] {p.name}  → vib={vib}")

        # vib が同じなら最初のファイルを採用（中身は同じ前提）
        if vib not in vib_sources:
            vib_sources[vib] = (prefix, p)

    # -------------------------------------------
    # vib ごとに25ファイル生成
    # -------------------------------------------
    for vib, (prefix, src_path) in vib_sources.items():
        print(f"\n[PROCESS vib={vib}] source = {src_path.name}")

        suffix = src_path.suffix

        for A_new in range(1, 6):
            for B_new in range(1, 6):
                new_name = f"{prefix}_{A_new}_{B_new}_{vib}{suffix}"
                out_path = outdir / new_name

                shutil.copy(src_path, out_path)
                print(f"  -> {new_name}")

    print("\n[DONE] 全ての vib について A×B=25 個ずつ生成しました。")


if __name__ == "__main__":
    main()
