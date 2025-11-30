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

    # 数字になっている部分を探す（漢字などは無視）
    numeric_indices = []
    for i, p in enumerate(parts):
        try:
            int(p)
            numeric_indices.append(i)
        except:
            pass

    if len(numeric_indices) < 3:
        raise ValueError(f"File name does not contain 3 numeric parts: {stem}")

    A_idx, B_idx, vib_idx = numeric_indices[-3:]

    A = int(parts[A_idx])
    B = int(parts[B_idx])
    vib = int(parts[vib_idx])

    prefix = "_".join(parts[:A_idx])
    return prefix, A, B, vib


def main():
    parser = argparse.ArgumentParser(
        description="Generate SynthV files by replacing A,B values based on vib."
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

    vib_sources = {}   # vib → (prefix, A_orig, B_orig, src_path)

    # ================================
    # 入力ファイルの読み取り
    # ================================
    for audio_str in args.audio:
        p = Path(audio_str)
        prefix, A, B, vib = parse_synthv_threeparams(p.stem)

        print(f"[READ] {p.name}  → vib={vib}")

        if vib not in vib_sources:
            vib_sources[vib] = (prefix, A, B, p)

    # ================================
    # vib ごとに生成
    # ================================
    for vib, (prefix, A_orig, B_orig, src_path) in vib_sources.items():
        suffix = src_path.suffix
        print(f"\n[PROCESS vib={vib}] source = {src_path.name}")

        # ----------------------------
        # ★ 花隈AIの特別ルール
        # ----------------------------
        if "花隈AI" in prefix:
            if A_orig == 1:
                A_range = [1, 2, 3]
            elif A_orig == 4:
                A_range = [4, 5]
            else:
                # 1 と 4 以外は通常ルールに落とす
                A_range = range(1, 6)
        else:
            # 通常AI → A=1..5
            A_range = range(1, 6)

        # B は全 AI 共通で 1..5
        B_range = range(1, 6)

        # ----------------------------
        # A_new × B_new の組み合わせ生成
        # ----------------------------
        for A_new in A_range:
            for B_new in B_range:
                new_name = f"{prefix}_{A_new}_{B_new}_{vib}{suffix}"
                out_path = outdir / new_name
                shutil.copy(src_path, out_path)
                print(f"  -> {new_name}")

    print("\n[DONE] 全てのファイルの複製を完了しました。")


if __name__ == "__main__":
    main()

