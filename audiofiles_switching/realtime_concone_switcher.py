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
- すべての WAV（メトロノーム／伴奏／AI 候補）は事前読み込みし、再生時のラグをゼロに近づける
- 現在時刻（メトロノーム開始 0 秒からの秒数）と再生中のファイル名を
  別スレッドでリアルタイム表示する
"""

import argparse
import os
import csv
import time
import threading
from typing import List, Tuple, Dict

import numpy as np
import sounddevice as sd
import soundfile as sf

import real_file_selector as rfs

# ==============================
# グローバル（ログ用）
# ==============================
GLOBAL_START_TIME: float | None = None
CURRENT_PLAYING: str = "IDLE"
LOGGING_STOP: bool = False


def logging_worker(interval: float = 0.5) -> None:
    """
    メトロノーム再生開始時刻(0秒)からの経過時間と、
    現在再生中と宣言されているファイル名を定期的に表示する。
    """
    global GLOBAL_START_TIME, CURRENT_PLAYING, LOGGING_STOP
    while not LOGGING_STOP:
        if GLOBAL_START_TIME is not None:
            elapsed = time.time() - GLOBAL_START_TIME
            print(f"[TIME] {elapsed:8.3f} sec | playing: {CURRENT_PLAYING}")
        time.sleep(interval)


# --------------------------------------------
# 録音 / WAV 生成
# --------------------------------------------

def record_exact_time(start_sec: float, stop_sec: float,
                      sr: int, channels: int, device: int | None) -> np.ndarray:
    """
    start_sec〜stop_sec の区間だけ録音する。
    （ここでは「区間長 = stop_sec - start_sec」分だけ録音するだけで、
      絶対時刻への同期は行わない）
    """
    duration = stop_sec - start_sec
    print(f"[REC] {duration:.3f} 秒録音 (区間 {start_sec}-{stop_sec} sec)")

    # 録音側デバイス設定（入力・出力を同じ番号に）
    if device is not None:
        sd.default.device = (device, device)

    print(f"[REC] device={sd.default.device}, sr={sr}, channels={channels}")

    rec = sd.rec(
        int(duration * sr),
        samplerate=sr,
        channels=channels,
        dtype="float32",
    )
    sd.wait()
    print("[REC] 録音終了")
    print(f"[REC] rec.shape={rec.shape}, min={rec.min():.3f}, max={rec.max():.3f}")

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


# ============================================
# 事前読み込みヘルパ
# ============================================

def load_wav_mono_or_stereo(path: str, target_sr: int) -> tuple[np.ndarray, int]:
    """
    WAV を読み込み、target_sr と異なる場合はエラーにする。
    事前にすべて読み込んでおき、再生時のラグを減らす。
    """
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    y, fs = sf.read(path, always_2d=True)
    if fs != target_sr:
        raise ValueError(f"Sample rate mismatch: {path} (got {fs}, expected {target_sr})")

    return y.astype(np.float32), fs


def preload_ai_candidates(
    ai_root: str,
    base_prefix: str,
    oq_candidates: List[int],
    rq_candidates: List[int],
    max_seg_index: int,
    target_sr: int,
) -> Dict[str, np.ndarray]:
    """
    Q1 = YES: 全組み合わせのファイルが ai_root にある前提。
    Oq/Rq は与えられた候補、vib は 1〜5 を総当たりで事前読み込みしておく。
    存在しないファイルはスキップ。
    """
    cache: Dict[str, np.ndarray] = {}

    print("[PRELOAD] AI 音声ファイルを事前読み込み開始...")
    total_try = 0
    total_loaded = 0

    for seg_idx in range(1, max_seg_index + 1):
        for oq in oq_candidates:
            for rq in rq_candidates:
                for vib in range(1, 6):
                    fname = rfs.build_audio_filename(
                        base_prefix=base_prefix,
                        oq_idx=oq,
                        rq_idx=rq,
                        vib_idx=vib,
                        segment_index=seg_idx,
                        ext=".wav",
                    )
                    path = os.path.join(ai_root, fname)
                    total_try += 1
                    if not os.path.exists(path):
                        continue
                    try:
                        y, fs = sf.read(path, always_2d=True)
                        if fs != target_sr:
                            raise ValueError(
                                f"Sample rate mismatch: {path} (got {fs}, expected {target_sr})"
                            )
                        cache[path] = y.astype(np.float32)
                        total_loaded += 1
                    except Exception as e:
                        print(f"[PRELOAD][WARN] {path} 読み込み失敗: {e}")

    print(f"[PRELOAD] 試行 {total_try} ファイル中 {total_loaded} ファイルをキャッシュ")
    return cache


# --------------------------------------------
# メイン処理
# --------------------------------------------

def main():
    global GLOBAL_START_TIME, CURRENT_PLAYING, LOGGING_STOP

    parser = argparse.ArgumentParser(description="Real-time Concone switcher with segment ranges")

    # ===== モデル =====
    parser.add_argument("--percentiles_csv", type=str, required=True)
    parser.add_argument("--extended_csv",    type=str, required=True)
    parser.add_argument("--reg_spline_csv",  type=str, required=True)
    parser.add_argument("--gmm_csv",         type=str, required=True)

    parser.add_argument("--segment_id", type=str, default="segALL")
    parser.add_argument("--quality",    type=str, default="CLEAN")

    # ===== パラメータ候補 =====
    parser.add_argument("--oq_candidates", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    parser.add_argument("--rq_candidates", type=int, nargs="+", default=[1, 2, 3, 4, 5])

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
    parser.add_argument("--sr", type=int, default=44100)
    parser.add_argument("--in_channels", type=int, default=1)
    parser.add_argument("--device", type=int, default=None)

    parser.add_argument("--metronome_wav", type=str, required=True,
                        help="最初のメトロノーム wav")
    parser.add_argument("--accomp_wav", type=str, default=None,
                        help="ピアノ伴奏 WAV。AI 音声と同時にミックス再生する")

    # ===== ログ =====
    parser.add_argument("--outdir", type=str, required=True)

    args = parser.parse_args()

    sr = args.sr

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
    seg_ranges: List[Tuple[float, float]] = []
    for s in args.segments:
        try:
            a, b = s.split("-")
            seg_ranges.append((float(a), float(b)))
        except Exception:
            raise ValueError(f"--segments の形式が不正です: {s}")

    if len(seg_ranges) == 0:
        raise ValueError("--segments が空です。例: --segments \"6-12\" \"12-18\"")

    # ===== メトロノーム事前読み込み =====
    print(f"[LOAD] メトロノーム: {args.metronome_wav}")
    metronome_audio, fs_click = load_wav_mono_or_stereo(args.metronome_wav, target_sr=sr)
    metronome_duration = metronome_audio.shape[0] / sr

    # ===== 伴奏事前読み込み =====
    accomp_audio: np.ndarray | None = None
    if args.accomp_wav is not None:
        print(f"[LOAD] 伴奏: {args.accomp_wav}")
        accomp_audio, fs_acc = load_wav_mono_or_stereo(args.accomp_wav, target_sr=sr)

    # ===== AI音声 事前読み込み =====
    max_seg_index = len(seg_ranges)
    ai_cache = preload_ai_candidates(
        ai_root=args.ai_root,
        base_prefix=args.base_prefix,
        oq_candidates=args.oq_candidates,
        rq_candidates=args.rq_candidates,
        max_seg_index=max_seg_index,
        target_sr=sr,
    )

    # ===== ログ初期化 =====
    with open(log_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["seg", "start", "end", "record_path",
                         "oq_idx", "rq_idx", "vib_idx", "ai_file"])

    # ===== ロギングスレッド開始 =====
    GLOBAL_START_TIME = time.time()
    CURRENT_PLAYING = "IDLE"
    LOGGING_STOP = False
    log_thread = threading.Thread(target=logging_worker, daemon=True)
    log_thread.start()

    # ===== メトロノーム再生 =====
    first_start = seg_ranges[0][0]  # 例: 6.0
    # first_start と metronome_duration は一致していることが望ましいが、
    # 一致していなくても、とりあえずメトロノーム WAV 全体を再生する。
    CURRENT_PLAYING = f"METRONOME: {os.path.basename(args.metronome_wav)}"
    print(f"[CLICK] メトロノーム再生開始 (duration={metronome_duration:.3f} sec)")
    if args.device is not None:
        sd.default.device = (args.device, args.device)
    sd.play(metronome_audio, samplerate=sr)
    sd.wait()
    print("[CLICK] メトロノーム再生終了")

    # ===== メインループ =====
    for i, (st, ed) in enumerate(seg_ranges, start=1):
        print("=" * 40)
        print(f"[SEG {i}] {st}-{ed} sec 録音開始")
        
        CURRENT_PLAYING = f"REC_SEG_{i}"

        # 録音
        rec = record_exact_time(st, ed, sr, args.in_channels, args.device)
        rec_wav = write_temp_wav(rec, sr, args.outdir, i)

        # 次セグメント
        next_seg = i + 1 if i < len(seg_ranges) else i

        if i == len(seg_ranges):
            # 最終セグメントは推定しない（AI 再生もなし）
            oq = rq = vib = -1
            ai_path = ""
        else:
            # AI ファイル選択
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

            # ===== 再生（AI + 伴奏ミックス） =====
            if ai_path in ai_cache:
                ai_audio = ai_cache[ai_path]
                # デバイス設定（出力）
                if args.device is not None:
                    sd.default.device = (args.device, args.device)

                if accomp_audio is None:
                    # 伴奏なし → AI のみ
                    CURRENT_PLAYING = f"AI: {os.path.basename(ai_path)} (no accomp)"
                    sd.play(ai_audio, samplerate=sr)
                    sd.wait()
                else:
                    # seg の長さを AI に合わせる
                    seg_len = ai_audio.shape[0]

                    # 伴奏が seg_len より短い場合はループ
                    if accomp_audio.shape[0] < seg_len:
                        rep = int(np.ceil(seg_len / accomp_audio.shape[0]))
                        acomp_seg = np.tile(accomp_audio, (rep, 1))[:seg_len]
                    else:
                        acomp_seg = accomp_audio[:seg_len]

                    mix = ai_audio.astype(np.float32) + acomp_seg.astype(np.float32)
                    maxv = np.max(np.abs(mix))
                    if maxv > 1.0:
                        mix = mix / maxv

                    CURRENT_PLAYING = (
                        f"AI: {os.path.basename(ai_path)} + ACCOMP: {os.path.basename(args.accomp_wav)}"
                    )
                    sd.play(mix, samplerate=sr)
                    sd.wait()
            else:
                print(f"[WARN] AI file not preloaded or missing: {ai_path}")
                oq = rq = vib = -1  # 一応異常値

        # ログ追記
        with open(log_csv, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([i, st, ed, rec_wav, oq, rq, vib, ai_path])

    print("[DONE] 全セグメント処理完了")

    # ===== ロギングスレッド終了 =====
    CURRENT_PLAYING = "IDLE"
    LOGGING_STOP = True
    # デーモンスレッドなので join は必須ではないが、念のため
    time.sleep(0.2)


if __name__ == "__main__":
    main()
