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
    区間録音を outdir に保存して実ファイルパスを返す
    """
    if y.ndim > 1:
        y_mono = y.mean(axis=1)
    else:
        y_mono = y

    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, f"record_stream_seg{seg_index:02d}.wav")
    sf.write(path, y_mono, sr)
    return path


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
) -> Tuple[int, int, int, str, dict]:
    """
    real_file_selector.py のコアを使って、
    入力 wav から (Oq_idx, Rq_idx, vib_idx, 次セグメント用ファイルパス) を選ぶ。
    :contentReference[oaicite:3]{index=3}
    """
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

    ai_path = os.path.join(ai_root, fname)

    debug = {
        "features": feat,
        "oqrq": debug_oqrq,
        "vib": debug_vib,
    }

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
    def __init__(self, start: float, end: float):
        self.start = float(start)
        self.end = float(end)
        # 「終了時刻-1sec」で打ち切り（seg の長さが 1 秒以下の場合の安全対策）
        self.cut_time = max(self.start, self.end - 1.0)

        self.ai_path: Optional[str] = None
        self.ai_audio: Optional[np.ndarray] = None
        self.oq: Optional[int] = None
        self.rq: Optional[int] = None
        self.vib: Optional[int] = None


# ============================================
# 解析スレッド（録音された区間から次セグメントの Saki を決める）
# ============================================

def analysis_worker(
    sr: int,
    mic_buffer: np.ndarray,
    seg_configs: List[SegmentConfig],
    outdir: str,
    reg_bank: rfs.RegressionBank,
    perc_bank: rfs.PercentileBank,
    gmm_bank: rfs.GMMParams,
    oq_candidates: List[int],
    rq_candidates: List[int],
    base_prefix: str,
    ai_root: str,
    ai_cache: Dict[str, np.ndarray],
    analysis_channel: int,
):
    """
    区間 i の [start_i, end_i-1sec] 録音を使って、
    区間 i+1 で流す Saki を決める。
    """
    global GLOBAL_SAMPLE_IDX

    num_segments = len(seg_configs)

    for seg_idx in range(num_segments - 1):  # 最後の seg は次がない
        seg_num = seg_idx + 1  # 1-based 表示用
        cfg = seg_configs[seg_idx]
        next_cfg = seg_configs[seg_idx + 1]

        cut_time = cfg.cut_time
        print(f"[ANALYSIS] seg{seg_num:02d}: cut_time={cut_time:.3f}s (end={cfg.end:.3f}s)")

        # 解析に使う時刻まで待機
        while GLOBAL_SAMPLE_IDX / sr < cut_time:
            time.sleep(0.05)

        # バッファ範囲を計算
        start_sample = int(cfg.start * sr)
        end_sample = int(cfg.cut_time * sr)

        if end_sample <= start_sample:
            print(f"[ANALYSIS][WARN] seg{seg_num:02d} 有効サンプルがありません (start={start_sample}, end={end_sample})")
            continue

        # 録音波形をコピー
        seg_data = mic_buffer[start_sample:end_sample, :]
        if seg_data.size == 0:
            print(f"[ANALYSIS][WARN] seg{seg_num:02d} 録音データが空です")
            continue

        # 1ch 目だけ（仮。将来 multi-channel に拡張可）
        analysis_ch = args.analysis_channel
        if analysis_ch < 0 or analysis_ch >= seg_data.shape[1]:
            raise ValueError(f"analysis_channel={analysis_ch} が入力チャンネル数 {seg_data.shape[1]} を超えています")

        y = seg_data[:, analysis_ch]


        # 一時 wav に書き出し
        rec_wav = write_temp_wav(y, sr, outdir, seg_num)

        # real_file_selector で次セグメント用ファイルを決定
        next_seg_num = seg_idx + 2  # seg_idx+1 の次
        oq, rq, vib, ai_path, debug = select_ai_file(
            wav_path=rec_wav,
            reg_bank=reg_bank,
            perc_bank=perc_bank,
            gmm_bank=gmm_bank,
            oq_cand=oq_candidates,
            rq_cand=rq_candidates,
            base_prefix=base_prefix,
            next_seg=next_seg_num,
            ai_root=ai_root,
        )

        # キャッシュにあるか確認
        if ai_path not in ai_cache:
            print(f"[ANALYSIS][WARN] seg{seg_num:02d} -> {ai_path} (キャッシュに存在しません)")
            continue

        next_cfg.ai_path = ai_path
        next_cfg.ai_audio = ai_cache[ai_path]
        next_cfg.oq = oq
        next_cfg.rq = rq
        next_cfg.vib = vib

        print(f"[ANALYSIS] seg{seg_num:02d} → seg{next_seg_num:02d}: "
              f"Oq={oq}, Rq={rq}, vib={vib}, file={os.path.basename(ai_path)}")


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

    # ===== AI音声ファイルの検索根 =====
    parser.add_argument("--ai_root", type=str, required=True)

    # ===== ベース名 =====
    parser.add_argument("--base_prefix", type=str, required=True)

    # ===== 録音・再生 =====
    parser.add_argument("--sr", type=int, default=44100)
    parser.add_argument("--in_channels", type=int, default=1)
    parser.add_argument("--analysis_channel", type=int, required=True,
                    help="real_file_selector で解析に使う入力チャンネル (0-based)")
    parser.add_argument("--input_device", type=int, required=True,
                    help="録音に使うデバイス index")
    parser.add_argument("--output_device", type=int, required=True,
                    help="再生に使うデバイス index（全て Focusrite 経由でOK）")
    parser.add_argument("--use_asio", action="store_true",
                    help="Use ASIO driver (recommended for Focusrite)")
    # parser.add_argument セクションに追記
    parser.add_argument("--use_wasapi", action="store_true",
                    help="Use Windows WASAPI host API (for Focusrite if ASIO is unavailable)")

    parser.add_argument("--blocksize", type=int, default=256)

    parser.add_argument("--metronome_wav", type=str, required=True,
                        help="最初のメトロノーム wav")
    parser.add_argument("--accomp_wav", type=str, required=True,
                        help="ピアノ伴奏 WAV。常にバックで再生する")

    # ===== ログ =====
    parser.add_argument("--outdir", type=str, required=True)

    args = parser.parse_args()

    sr = args.sr
    SR_GLOBAL = sr
    
    # args = parser.parse_args() の直後に追加
    if args.use_asio and args.use_wasapi:
        raise ValueError("--use_asio と --use_wasapi は同時に指定できません")

        # WASAPI ホストAPIに切り替え
        if args.use_wasapi:
            wasapi_index = None
            for idx, api in enumerate(sd.query_hostapis()):
                if api["name"] == "Windows WASAPI":
                    wasapi_index = idx
                    break
            if wasapi_index is None:
                raise RuntimeError("WASAPI が見つかりませんでした")
            sd.default.hostapi = wasapi_index
            print(f"[INFO] ホストAPIを WASAPI に設定しました (index={wasapi_index})")


    # ===== 構造準備 =====
    os.makedirs(args.outdir, exist_ok=True)
    log_csv = os.path.join(args.outdir, "realtime_stream_log.csv")

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

    # SegmentConfig を作成
    seg_configs: List[SegmentConfig] = [
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

    # ===== 解析スレッド開始 =====
    analysis_thread = threading.Thread(
        target=analysis_worker,
        args=(
            sr,
            mic_buffer,
            seg_configs,
            args.outdir,
            reg_bank,
            perc_bank,
            gmm_bank,
            args.oq_candidates,
            args.rq_candidates,
            args.base_prefix,
            args.ai_root,
            ai_cache,
            args.analysis_channel, 
        ),
        daemon=True,
    )
    analysis_thread.start()

    # ===== Streaming callback =====
    current_seg_idx = -1  # -1: still before first segment

    def audio_callback(indata, outdata, frames, time_info, status):
        nonlocal current_seg_idx
        global GLOBAL_SAMPLE_IDX, CURRENT_PLAYING
        
        t = GLOBAL_SAMPLE_IDX / SR_GLOBAL
        if t < 10.0 and frames > 0:
            levels = [float(np.max(np.abs(indata[:, ch]))) for ch in range(indata.shape[1])]
            min_in = float(np.min(indata))
            max_in = float(np.max(indata))
            print(f"[DEBUG-REC] ch_levels={levels}")
            print(f"[DEBUG-REC] indata: min={min_in:.3f}, max={max_in:.3f}")


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
                    CURRENT_PLAYING = "ACCOMP_ONLY"
            else:
                CURRENT_PLAYING = "ACCOMP_ONLY"

        outdata[:] = out
        GLOBAL_SAMPLE_IDX += frames

    # ===== Streaming 実行 =====
    print("[STREAM] Start full-duplex stream (WDM-KS, int16)")

    # WDM-KS では extra_settings / WASAPI 設定は不要

    # ===== デバイス情報を確認 =====
    try:
        dev_in = sd.query_devices(args.input_device)
        dev_out = sd.query_devices(args.output_device)
        print(f"[INFO] WDM-KS input_device={args.input_device} "
              f"(max_in={dev_in['max_input_channels']})")
        print(f"[INFO] WDM-KS output_device={args.output_device} "
              f"(max_out={dev_out['max_output_channels']})")
    except Exception as e:
        print("[WARN] Failed to read device info:", e)

    # ===== ストリーム中のループ =====
    def run_stream_loop():
        while GLOBAL_SAMPLE_IDX / sr < total_duration:
            time.sleep(0.1)

    # ===== int16 入出力に対応したコールバック =====
    def ks_callback(indata, outdata, frames, time_info, status):
        global GLOBAL_SAMPLE_IDX
        global CURRENT_PLAYING
        nonlocal current_seg_idx

        # ---- int16 → float32 に変換 ----
        in_float = indata.astype(np.float32) / 32768.0

        # ---- 録音データを mic_buffer に保存 ----
        start_idx = GLOBAL_SAMPLE_IDX
        if start_idx < mic_buffer.shape[0]:
            write_len = min(frames, mic_buffer.shape[0] - start_idx)
            mic_buffer[start_idx:start_idx + write_len, :in_float.shape[1]] = in_float[:write_len]

        # ---- 出力用 float32 バッファ ----
        out = np.zeros((frames, 2), dtype=np.float32)

        # ===== フレームごとに処理 =====
        for n in range(frames):
            t = (GLOBAL_SAMPLE_IDX + n) / sr

            # --- メトロノーム ---
            if t < metronome_duration:
                mi = int(t * sr)
                if 0 <= mi < metronome_audio.shape[0]:
                    out[n, 0] += metronome_audio[mi, 0]
                    out[n, 1] += metronome_audio[mi, 0]
                CURRENT_PLAYING = "METRONOME"
                continue

            # --- 伴奏 ---
            acc_t = t - metronome_duration
            ai = int(acc_t * sr) % accomp_audio.shape[0]
            out[n] += accomp_audio[ai, :2]

            # --- セグメント遷移 ---
            if 0 <= current_seg_idx < num_segments:
                if t >= seg_configs[current_seg_idx].end:
                    current_seg_idx += 1

            if current_seg_idx < 0 and t >= seg_configs[0].start:
                current_seg_idx = 0

            # --- Saki 再生 ---
            if 0 <= current_seg_idx < num_segments:
                cfg = seg_configs[current_seg_idx]
                if cfg.ai_audio is not None and cfg.start <= t < cfg.end:
                    off = int((t - cfg.start) * sr)
                    if 0 <= off < cfg.ai_audio.shape[0]:
                        out[n] += cfg.ai_audio[off, :2]
                        CURRENT_PLAYING = os.path.basename(cfg.ai_path)
                else:
                    CURRENT_PLAYING = "ACCOMP_ONLY"

        # 次のブロックへ
        GLOBAL_SAMPLE_IDX += frames

        # ---- float32 → int16 変換して出力 ----
        out16_full = np.zeros((frames, 8), dtype=np.int16)
        out16_full[:, 0:2] = out16   # あなたの既存2ch出力
        outdata[:] = out16_full



    # ===== Stream パラメータ =====
    stream_args = {
        "samplerate": sr,
        "blocksize": args.blocksize,
        "channels": (args.in_channels, 8),  # 入力8ch, 出力8ch
        "callback": ks_callback,
        "device": (args.input_device, args.output_device),
        "dtype": "int16",  # ← ここが重要
    }

    print("[INFO] opening WDM-KS stream with:", stream_args)

    try:
        with sd.Stream(**stream_args):
            run_stream_loop()
    except Exception as e:
        print("[ERROR] Failed to start WDM-KS stream:", e)
        return

    print("[STREAM] Finished (WDM-KS)")





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


if __name__ == "__main__":
    main()
