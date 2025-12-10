#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
realtime_concone_streaming.py

Aパターン：
- 伴奏は常に流し続ける（全区間を通してループ）
- 各セグメントごとに Saki AI のファイルを切り替える
- 録音と再生は同時（フルデュプレックス）
- 区間 i の録音 [start_i, end_i - 1sec] を使って「次のセグメント i+1」で流す Saki を予測
- seg1 だけはデフォルトの (Oq,Rq,vib) で開始

前の realtime_concone_switcher.py を
  - sd.play/sd.wait ベース → sd.Stream ベース
に書き換えた streaming 版。:contentReference[oaicite:1]{index=1}

real_file_selector のコア部（RegressionBank, PercentileBank, select_oq_rq_indices など）は
real_file_selector.py にあるものを利用する。:contentReference[oaicite:2]{index=2}
"""

import argparse
import os
import csv
import time
import threading
from typing import List, Tuple, Dict, Optional
import traceback

# === Multiprocessing for heavy feature analysis ===
from multiprocessing import Process, Queue


import numpy as np
import sounddevice as sd
import soundfile as sf

import real_file_selector as rfs

# ==============================
# グローバル（ログ用）
# ==============================
GLOBAL_SAMPLE_IDX: int = 0
SR_GLOBAL: int = 44100
CURRENT_PLAYING: str = "IDLE"
LOGGING_STOP: bool = False


def logging_worker(interval: float = 0.5) -> None:
    """
    メトロノーム開始時刻(0秒)からの経過時間と、
    現在再生中ファイル名を定期的に表示する。
    """
    global GLOBAL_SAMPLE_IDX, SR_GLOBAL, CURRENT_PLAYING, LOGGING_STOP
    while not LOGGING_STOP:
        elapsed = GLOBAL_SAMPLE_IDX / SR_GLOBAL
        print(f"[TIME] {elapsed:8.3f} sec | playing: {CURRENT_PLAYING}")
        time.sleep(interval)


# --------------------------------------------
# 録音 WAV 一時保存
# --------------------------------------------

def write_temp_wav(y: np.ndarray, sr: int, outdir: str, seg_index: int) -> str:
    """
    WAV を安全に保存し、他プロセスが読む前に
    'data チャンク未完成' を防ぐ
    """
    if y.ndim > 1:
        y_mono = y.mean(axis=1)
    else:
        y_mono = y

    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, f"record_stream_seg{seg_index:02d}.wav")

    # ★ soundfile のファイルオブジェクトを明示的に close させる
    with sf.SoundFile(path, mode="w", samplerate=sr, channels=1, format="WAV") as f:
        f.write(y_mono.astype(np.float32))

    # ★ Windows の場合、書き込みバッファがすぐに解放されないことがある → 明示 wait
    time.sleep(0.01)

    # ★ ファイルサイズ確認（壊れていれば size が極端に小さい）
    if os.path.getsize(path) < 44:  # WAV header より小さい
        print(f"[WRITE][ERROR] WAV corrupted: {path}")

    return path

def safety_wait_for_wav(path, timeout=0.05):
    """
    WAV の 'data' チャンクが完成するまで待つ。
    Windows + マルチプロセスでは重要。
    """
    import os, time

    start = time.time()
    last_size = -1

    while time.time() - start < timeout:
        if not os.path.exists(path):
            time.sleep(0.005)
            continue

        size = os.path.getsize(path)
        if size > 44 and size == last_size:
            return True  # 44byte は WAV ヘッダ最小サイズ
        last_size = size
        time.sleep(0.005)

    print(f"[WARN] WAV not stable yet: {path}")
    return False


# --------------------------------------------
# AI音声ファイル決定（real_file_selector のラッパ）
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
    seg_label: str = "UNKNOWN",
) -> Tuple[int, int, int, str, dict]:
    """
    real_file_selector.py のコアを使って、
    入力 wav から (Oq_idx, Rq_idx, vib_idx, 次セグメント用ファイルパス) を選ぶ。

    ★ デバッグ用に、各ステップごとに必ずログを出す。
    """

    print(f"[SELECT] {seg_label}: select_ai_file START wav_path={wav_path}")
    
    
    # ★ WAV 完成チェック（最大 50ms）
    if not safety_wait_for_wav(wav_path):
        raise RuntimeError(f"WAV not ready: {wav_path}")

    try:
        # ---- Step 1: 特徴量抽出 ----
        print(f"[SELECT] {seg_label}: Step 1/4 extract_acoustic_features 開始")
        feat = rfs.extract_acoustic_features(wav_path)
        print(f"[SELECT] {seg_label}: Step 1/4 完了 feature_keys={list(feat.keys())}")

        # ---- Step 2: Oq/Rq 推定 ----
        print(f"[SELECT] {seg_label}: Step 2/4 select_oq_rq_indices 開始")
        oq_idx, rq_idx, debug_oqrq = rfs.select_oq_rq_indices(
            features=feat,
            reg_bank=reg_bank,
            perc_bank=perc_bank,
            gmm_bank=gmm_bank,
            oq_candidates=oq_cand,
            rq_candidates=rq_cand,
        )
        print(f"[SELECT] {seg_label}: Step 2/4 完了 Oq_idx={oq_idx}, Rq_idx={rq_idx}")

        # ---- Step 3: vibrato 推定 ----
        print(f"[SELECT] {seg_label}: Step 3/4 select_vibrato_index_from_percentiles 開始")
        vib_idx, debug_vib = rfs.select_vibrato_index_from_percentiles(
            vibrato_cents_abs=feat["vibrato_cents_abs"],
            perc_bank=perc_bank,
        )
        print(f"[SELECT] {seg_label}: Step 3/4 完了 vib_idx={vib_idx}")

        # ---- Step 4: ファイル名組み立て ----
        print(f"[SELECT] {seg_label}: Step 4/4 build_audio_filename 開始")
        fname = rfs.build_audio_filename(
            base_prefix=base_prefix,
            oq_idx=oq_idx,
            rq_idx=rq_idx,
            vib_idx=vib_idx,
            segment_index=next_seg,
            ext=".wav",
        )
        ai_path = os.path.join(ai_root, fname)
        print(f"[SELECT] {seg_label}: Step 4/4 完了 ai_path={ai_path}")

    except Exception as e:
        # ★ここで必ず、どこで死んだかのステップ名と例外内容を出す
        print(f"[SELECT][ERROR] {seg_label}: select_ai_file 内で例外発生: {e!r}")
        traceback.print_exc()
        # analysis_worker 側にも伝えるため再スロー
        raise

    debug = {
        "features": feat,
        "oqrq": debug_oqrq,
        "vib": debug_vib,
    }

    print(
        f"[SELECT] {seg_label}: select_ai_file DONE "
        f"(Oq={oq_idx}, Rq={rq_idx}, vib={vib_idx}, file={ai_path})"
    )
    return oq_idx, rq_idx, vib_idx, ai_path, debug



# ============================================
# WAV ロード & 事前キャッシュ
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
    全候補(Saki)を事前読み込みしておく。:contentReference[oaicite:4]{index=4}

    - segment 1..max_seg_index
    - Oq 候補 × Rq 候補 × vib=1..5
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


# ============================================
# Streaming 用セグメント設定
# ============================================

class SegmentConfig:
    def __init__(self, start: float, end: float, cut_margin: float = 2.0):
        self.start = float(start)
        self.end = float(end)
        self.cut_time = max(self.start, self.end - cut_margin)

        self.ai_path = None
        self.ai_audio = None
        self.oq = None
        self.rq = None
        self.vib = None
        self.analysis_sent = False




# ============================================
# 解析スレッド（録音された区間から次セグメントの Saki を決める）
# ============================================

# ============================================
# Multiprocessing 版 解析ワーカー
# ============================================

def analysis_worker_proc(
    task_queue: Queue,
    result_queue: Queue,
    reg_bank,
    perc_bank,
    gmm_bank,
    oq_candidates,
    rq_candidates,
    base_prefix,
    ai_root,
):
    """
    別プロセスで動く解析ワーカー。

    task_queue に (seg_num, wav_path, next_seg_num) が来たら、
    real_file_selector を実行し、結果を result_queue に入れる。
    """

    while True:
        item = task_queue.get()
        if item is None:
            break

        seg_num, wav_path, next_seg = item

        try:
            oq, rq, vib, ai_path, debug = select_ai_file(
                wav_path=wav_path,
                reg_bank=reg_bank,
                perc_bank=perc_bank,
                gmm_bank=gmm_bank,
                oq_cand=oq_candidates,
                rq_cand=rq_candidates,
                base_prefix=base_prefix,
                next_seg=next_seg,
                ai_root=ai_root,
                seg_label=f"seg{seg_num:02d}",
            )
            
            # ★ ここで次のAI音声を読み込んで numpy にしてしまう（事前ロード）
            ai_audio = None
            try:
                if os.path.exists(ai_path):
                    y_ai, fs_ai = sf.read(ai_path, always_2d=True)
                    ai_audio = y_ai.astype(np.float32)
            except Exception as e:
                print(f"[ANALYSIS-PROC][WARN] AI wav preload failed: {ai_path}: {e}")


            result_queue.put((seg_num, next_seg, oq, rq, vib, ai_path, ai_audio, debug))

        except Exception as e:
            print(f"[ANALYSIS-PROC][ERROR] seg{seg_num:02d}: {e}")
            traceback.print_exc()
            result_queue.put((seg_num, next_seg, None, None, None, None, None))





# ============================================
# メイン処理（Streaming）
# ============================================

def main():
    global GLOBAL_SAMPLE_IDX, SR_GLOBAL, CURRENT_PLAYING, LOGGING_STOP

    parser = argparse.ArgumentParser(description="Real-time Concone streaming switcher (full-duplex)")

    # ===== モデル =====
    parser.add_argument("--percentiles_csv", type=str, required=True)
    parser.add_argument("--extended_csv",    type=str, required=True)
    parser.add_argument("--reg_spline_csv",  type=str, required=True)
    parser.add_argument("--gmm_csv",         type=str, required=True)

    parser.add_argument("--segment_id", type=str, default="ALL")
    parser.add_argument("--quality",    type=str, default="CLEAN")

    # ===== パラメータ候補 =====
    parser.add_argument("--oq_candidates", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    parser.add_argument("--rq_candidates", type=int, nargs="+", default=[1, 2, 3, 4, 5])

    # ===== セグメント範囲 =====
    parser.add_argument(
        "--segments", nargs="+", required=True,
        help='例: --segments "6-12" "12-18" "18-24"'
    )
    
    parser.add_argument("--cut_margin", type=float, default=2.0,
                    help="Segment end minus this seconds for analysis cutoff (default: 2.0)")


    # ===== AI音声ファイルの検索根 =====
    parser.add_argument("--ai_root", type=str, required=True)

    # ===== ベース名 =====
    parser.add_argument("--base_prefix", type=str, required=True)

    # ===== 録音・再生 =====
    parser.add_argument("--sr", type=int, default=44100)
    parser.add_argument("--in_channels", type=int, default=1)
    parser.add_argument(
        "--analysis_channel",
        type=int,
        required=True,
        help="real_file_selector で解析に使う入力チャンネル (0-based)",
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

    parser.add_argument("--blocksize", type=int, default=256)

    parser.add_argument(
        "--metronome_wav",
        type=str,
        required=True,
        help="最初のメトロノーム wav",
    )
    parser.add_argument(
        "--accomp_wav",
        type=str,
        required=True,
        help="ピアノ伴奏 WAV。常にバックで再生する",
    )

    # ===== ログ =====
    parser.add_argument("--outdir", type=str, required=True)

    # ===== 引数確定 =====
    args = parser.parse_args()
    
    if args.segment_id.upper() == "ALL":
        seg_id_for_percentiles = "segALL"
    else:
        seg_id_for_percentiles = args.segment_id

    sr = args.sr
    SR_GLOBAL = sr



    # ===== 構造準備 =====
    os.makedirs(args.outdir, exist_ok=True)
    log_csv = os.path.join(args.outdir, "realtime_stream_log.csv")

    # ===== モデル読み込み =====
    reg_bank = rfs.RegressionBank(args.reg_spline_csv)
    perc_bank = rfs.PercentileBank(
        percentiles_csv=args.percentiles_csv,
        extended_csv=args.extended_csv,
        segment_id=seg_id_for_percentiles,
        quality=args.quality,
    )
    gmm_bank = rfs.load_gmm_params(args.gmm_csv)
    
    # ======================
    # librosa.pyin numba warmup
    # ======================
    print("[WARMUP] librosa.pyin warmup start...")
    import librosa
    import numpy as np

    dummy = np.random.randn(44100).astype(np.float32)
    try:
        _f0, _, _ = librosa.pyin(
            dummy,
            fmin=80, fmax=800,
            sr=sr,
            frame_length=1024,
            hop_length=256,
        )
        print("[WARMUP] librosa.pyin warmup done.")
    except Exception as e:
        print("[WARMUP][WARN] librosa.pyin warmup failed:", e)

    # ============================================
    # ★ 追加：spectral, MFCC, STFT warmup
    # ============================================
    print("[WARMUP] spectral + MFCC warmup start...")
    try:
        S = np.abs(librosa.stft(dummy, n_fft=1024, hop_length=256))
        _ = librosa.feature.spectral_centroid(S=S, sr=sr)
        _ = librosa.feature.spectral_rolloff(S=S, sr=sr)
        _ = librosa.feature.spectral_bandwidth(S=S, sr=sr)
        _ = librosa.feature.mfcc(y=dummy, sr=sr, n_mfcc=5)
        print("[WARMUP] spectral + MFCC warmup done.")
    except Exception as e:
        print("[WARMUP][WARN] spectral warmup failed:", e)



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

    # SegmentConfig を作成
    seg_configs = [
        SegmentConfig(start, end) for (start, end) in seg_ranges
    ]


    num_segments = len(seg_configs)

    # ===== メトロノーム事前読み込み =====
    print(f"[LOAD] メトロノーム: {args.metronome_wav}")
    metronome_audio, fs_click = load_wav_mono_or_stereo(args.metronome_wav, target_sr=sr)
    metronome_duration = metronome_audio.shape[0] / sr
    metronome_name = os.path.basename(args.metronome_wav)

    # ===== 伴奏事前読み込み =====
    print(f"[LOAD] 伴奏: {args.accomp_wav}")
    accomp_audio, fs_acc = load_wav_mono_or_stereo(args.accomp_wav, target_sr=sr)

    # ===== AI 音声 事前読み込み =====
    max_seg_index = num_segments
    ai_cache = preload_ai_candidates(
        ai_root=args.ai_root,
        base_prefix=args.base_prefix,
        oq_candidates=args.oq_candidates,
        rq_candidates=args.rq_candidates,
        max_seg_index=max_seg_index,
        target_sr=sr,
    )
    
    print(sd.query_hostapis())
    for i, d in enumerate(sd.query_devices()):
        print(i, d['name'], '| hostapi =', sd.query_hostapis()[d['hostapi']]['name'])


    # ===== seg1 のデフォルト Saki を決定（中庸Oq/Rq, vib=3 とする） =====
    oq_candidates_sorted = sorted(args.oq_candidates)
    rq_candidates_sorted = sorted(args.rq_candidates)
    default_oq = oq_candidates_sorted[len(oq_candidates_sorted) // 2]
    default_rq = rq_candidates_sorted[len(rq_candidates_sorted) // 2]
    default_vib = 3

    fname_seg1 = rfs.build_audio_filename(
        base_prefix=args.base_prefix,
        oq_idx=default_oq,
        rq_idx=default_rq,
        vib_idx=default_vib,
        segment_index=1,
        ext=".wav",
    )
    path_seg1 = os.path.join(args.ai_root, fname_seg1)
    if path_seg1 in ai_cache:
        seg_configs[0].ai_path = path_seg1
        seg_configs[0].ai_audio = ai_cache[path_seg1]
        seg_configs[0].oq = default_oq
        seg_configs[0].rq = default_rq
        seg_configs[0].vib = default_vib
        print(f"[INIT] seg01 default: {os.path.basename(path_seg1)} "
              f"(Oq={default_oq}, Rq={default_rq}, vib={default_vib})")
    else:
        print(f"[INIT][WARN] デフォルト seg1 ファイルがキャッシュにありません: {path_seg1}")

    # ===== マイク録音バッファ（全時間分を確保） =====
        # ===== 入力デバイスのチャンネル数に合わせて補正 =====
    try:
        dev_in = sd.query_devices(args.input_device)
        max_in_ch = dev_in["max_input_channels"]
        if args.in_channels > max_in_ch:
            print(f"[WARN] in_channels({args.in_channels}) > device max_input_channels({max_in_ch}) → {max_in_ch} に縮小します")
            args.in_channels = max_in_ch
    except Exception as e:
        print(f"[WARN] 入力デバイス情報取得に失敗しました: {e}")

    # ===== マイク録音バッファ（全時間分を確保）=====
    total_duration = seg_ranges[-1][1]
    total_samples = int((total_duration + 1.0) * sr)  # 余分に 1 秒
    mic_buffer = np.zeros((total_samples, args.in_channels), dtype=np.float32)


    # ===== ログ初期化（予測結果用）=====
    with open(log_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["seg", "start", "end", "cut_time",
                         "oq_idx", "rq_idx", "vib_idx", "ai_file"])

    # ===== ロギングスレッド開始 =====
    GLOBAL_SAMPLE_IDX = 0
    CURRENT_PLAYING = "IDLE"
    LOGGING_STOP = False
    log_thread = threading.Thread(target=logging_worker, daemon=True)
    log_thread.start()
    
    
        # ===== 解析プロセス準備 =====
    task_queue = Queue()
    result_queue = Queue()

    analysis_proc = Process(
        target=analysis_worker_proc,
        args=(
            task_queue,
            result_queue,
            reg_bank,
            perc_bank,
            gmm_bank,
            args.oq_candidates,
            args.rq_candidates,
            args.base_prefix,
            args.ai_root,
        ),
    )
    analysis_proc.start()
    print("[PROC] analysis process started")

    # ===== Streaming callback =====
    current_seg_idx = -1  # -1: still before first segment
    bad_seg_logged = [False] * num_segments
    
    last_debug_rec_time = -1.0

    def audio_callback(indata, outdata, frames, time_info, status):
        nonlocal current_seg_idx, last_debug_rec_time
        global GLOBAL_SAMPLE_IDX, CURRENT_PLAYING

        # ---- 録音レベル表示（0.5 秒周期・常時） ----
        if frames > 0:
            now_t = GLOBAL_SAMPLE_IDX / SR_GLOBAL

            # 直近のログから 0.5 秒以上経っていたら表示
            if (last_debug_rec_time < 0.0) or (now_t - last_debug_rec_time >= 0.5):
                levels = [
                    float(np.max(np.abs(indata[:, ch])))
                    for ch in range(indata.shape[1])
                ]
                min_in = float(np.min(indata))
                max_in = float(np.max(indata))

                print(f"[DEBUG-REC] t={now_t:.3f}s | ch_levels={levels}")
                print(f"[DEBUG-REC] indata: min={min_in:.3f}, max={max_in:.3f}")

                last_debug_rec_time = now_t




        if status:
            print(f"[SD STATUS] {status}", flush=True)

        # ---- 録音データを保存 ----
        start_idx = GLOBAL_SAMPLE_IDX
        end_idx = start_idx + frames
        if start_idx < mic_buffer.shape[0]:
            write_len = min(frames, mic_buffer.shape[0] - start_idx)
            mic_buffer[start_idx:start_idx + write_len, :indata.shape[1]] = indata[:write_len].astype(np.float32)


        # ---- 出力バッファを初期化 ----
        out = np.zeros((frames, 2), dtype=np.float32)


        for n in range(frames):
            t = (GLOBAL_SAMPLE_IDX + n) / sr

            # メトロノーム再生区間
            if t < metronome_duration:
                met_idx = int(t * sr)
                if 0 <= met_idx < metronome_audio.shape[0]:
                    out[n, 0] += metronome_audio[met_idx, 0]
                    out[n, 1] += metronome_audio[met_idx, 0]

                CURRENT_PLAYING = f"METRONOME: {metronome_name}"
                continue  # 伴奏/Saki はまだ鳴らさない

            # メトロノーム以降：伴奏 + 必要なら Saki
            # 伴奏（常にループ）
            # metronome_duration 経過後から伴奏スタート
            acc_t = t - metronome_duration
            acc_idx = int(acc_t * sr) % accomp_audio.shape[0]
            out[n, 0] += accomp_audio[acc_idx, 0]
            out[n, 1] += accomp_audio[acc_idx, 0]

            # 現在のセグメント index を更新（0-based）
            # t が seg_j.start〜seg_j.end の範囲にあれば j にいる
            # まず「今いる seg を抜けたか」をチェック
            if current_seg_idx >= 0 and current_seg_idx < num_segments:
                if t >= seg_configs[current_seg_idx].end:
                    current_seg_idx += 1
                    
            # === 解析依頼（cut_time到達で1回だけ） ===
            cfg = seg_configs[current_seg_idx]

            if (not cfg.analysis_sent) and (t >= cfg.cut_time):
                cfg.analysis_sent = True

                # WAV抽出
                seg_data = np.array(
                    mic_buffer[int(cfg.start * sr):int(cfg.cut_time * sr), args.analysis_channel],
                    dtype=np.float32
                ).copy()

                rec_wav = write_temp_wav(seg_data, sr, args.outdir, current_seg_idx+1)

                print(f"[PROC] send task seg{current_seg_idx+1:02d}")
                task_queue.put((current_seg_idx+1, rec_wav, current_seg_idx+2))



            # まだセグメントに入っていない場合、t が最初の seg に入ったら 0 に
            if current_seg_idx < 0:
                if t >= seg_configs[0].start:
                    current_seg_idx = 0

            # 現在のセグメントで Saki を再生
            if 0 <= current_seg_idx < num_segments:
                cfg = seg_configs[current_seg_idx]
                if cfg.ai_audio is not None and cfg.start <= t < cfg.end:
                    offset = int((t - cfg.start) * sr)
                    if 0 <= offset < cfg.ai_audio.shape[0]:
                        out[n, 0] += cfg.ai_audio[offset, 0]
                        out[n, 1] += cfg.ai_audio[offset, 0]
                        if cfg.ai_path is not None:
                            CURRENT_PLAYING = os.path.basename(cfg.ai_path)
                        else:
                            CURRENT_PLAYING = "ACCOMP_ONLY"
                else:
                    # このセグメントには ai_audio が設定されていない or 時間外
                    if 0 <= current_seg_idx < num_segments and not bad_seg_logged[current_seg_idx]:
                        print(
                            f"[DEBUG-PLAY] seg{current_seg_idx+1:02d}: "
                            f"ai_audio が None か、再生時間外のため ACCOMP_ONLY"
                        )
                        bad_seg_logged[current_seg_idx] = True
                    CURRENT_PLAYING = "ACCOMP_ONLY"
            else:
                CURRENT_PLAYING = "ACCOMP_ONLY"


        outdata[:] = out
        GLOBAL_SAMPLE_IDX += frames

    # ===== Streaming 実行（ASIO） =====
    print("[STREAM] Start full-duplex stream (ASIO)")

    # ===== デバイス情報を確認 =====
    try:
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
        print("[WARN] Failed to read ASIO device info:", e)

    # ===== ストリーム中のループ =====
    def run_stream_loop():
        while GLOBAL_SAMPLE_IDX / sr < total_duration:
            time.sleep(0.1)
            # === プロセスからの解析結果 ===
            while not result_queue.empty():
                seg_num, next_seg, oq, rq, vib, ai_path, ai_audio, debug = result_queue.get()

                if ai_path is None:
                    print(f"[PROC-RESULT] seg{seg_num:02d}: analysis failed")
                    continue

                # next_cfg に反映
                cfg_next = seg_configs[next_seg - 1]
                cfg_next.oq = oq
                cfg_next.rq = rq
                cfg_next.vib = vib
                cfg_next.ai_path = ai_path

                # キャッシュロード
                # ai_audio が analysis_worker で読み込み済みならそのまま使用
                if ai_audio is not None:
                    ai_cache[ai_path] = ai_audio

                # 念のため、キャッシュになければ読み込む fallback
                if ai_path not in ai_cache and os.path.exists(ai_path):
                    y_ai, fs_ai = sf.read(ai_path, always_2d=True)
                    ai_cache[ai_path] = y_ai.astype(np.float32)

                cfg_next.ai_audio = ai_cache.get(ai_path)

                print(f"[PROC-RESULT] seg{seg_num:02d} → seg{next_seg:02d}: {ai_path}")


    # ===== Stream パラメータ（ASIO 用）=====
    #  入力: in_channels（今回は 8ch）
    #  出力: 2ch（L/R）だけを使う
    stream_args = {
        "samplerate": sr,
        "blocksize": args.blocksize,
        "channels": (args.in_channels, 2),
        "callback": audio_callback,
        "device": (args.input_device, args.output_device),
        "dtype": "float32",
    }

    print("[INFO] opening ASIO stream with:", stream_args)

    try:
        with sd.Stream(**stream_args):
            run_stream_loop()
    except Exception as e:
        print("[ERROR] Failed to start ASIO stream:", e)
        return

    print("[STREAM] Finished (ASIO)")
    
    # ===== 解析スレッド終了を待つ（daemonだが一応少し待つ）=====
    time.sleep(1.0)

    # ===== 解析結果をログ CSV に書き出し =====
    with open(log_csv, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        for i, cfg in enumerate(seg_configs, start=1):
            writer.writerow([
                i,
                cfg.start,
                cfg.end,
                cfg.cut_time,
                cfg.oq if cfg.oq is not None else -1,
                cfg.rq if cfg.rq is not None else -1,
                cfg.vib if cfg.vib is not None else -1,
                cfg.ai_path if cfg.ai_path is not None else "",
            ])

    print("[DONE] 全セグメント処理完了")

    # ===== ロギングスレッド終了 =====
    CURRENT_PLAYING = "IDLE"
    LOGGING_STOP = True
    time.sleep(0.2)

import multiprocessing
multiprocessing.set_start_method("spawn", force=True)


if __name__ == "__main__":
    main()
