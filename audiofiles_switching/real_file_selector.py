#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
real_file_selector.py

コンコーネ歌唱の「現在の音響特徴量」から，Oq / Rq / vibrato パーセンタイルの
最尤組み合わせを推定し，それに対応する音声ファイル名を返すためのコア実装。

ポイント:
- 音響特徴量は GMM_additional 系とほぼ同じ set（ただし簡略版）
  - f0_hz
  - spec_centroid_hz
  - spec_bandwidth_hz
  - zcr
  - rms_dbfs
  - mfcc1〜mfcc5
  - vibrato_cents_abs
  - （CPP はダミー実装: 0.0。必要ならあとで書き換え）

- Δ特徴量（一次差分）は一切計算しない（削除済み）
  → 「パーセンタイルの変化速度」を使った誤差はここでは扱わない。

- Oq / Rq の誤差:
  - regression_spline_csv の spline モデルから「Oq 推定値」「Rq 推定値」を出す
  - パーセンタイル CSV( param_percentiles ) から Oq / Rq の目標値を取る
  - extended_stats の std を使って z-score の二乗 = ((pred - target)/std)^2
  - それを各パーセンタイル候補で計算して最小のものを選ぶ

- vibrato:
  - vibrato_cents_abs をそのまま使う
    「この vibrato は 1〜5 のどの bin に近いか」を決める
  - 誤差は (measured - center_of_bin)**2 で最小の番号を選ぶ

このスクリプトは「音響特徴量 → (Oq_idx, Rq_idx, vib_idx, ファイル名候補)」
の決定までを行う「コア」であり，
実際のリアルタイム録音・再生は multichannel_audio4.py などから呼び出して使う想定。

Usage 例（オフラインで wav 一個から試す）:
    python real_file_selector.py \
        --wav in.wav \
        --percentiles_csv merged_pre_id1-id2_solo_segALL_param_percentiles.csv \
        --extended_csv    merged_pre_id1-id2_solo_segALL_extended_stats.csv \
        --reg_spline_csv  merged_pre_id1-id2_solo_segALL_CLEAN_egg_vs_audio_regressions_spline.csv \
        --segment_id ALL \
        --quality CLEAN \
        --oq_candidates 1 2 3 4 5 \
        --rq_candidates 1 2 3 4 5 \
        --base_prefix concone_synthesizerV_SakiAI \
        --segment_index 10

"""

import argparse
import base64
import json
import pickle
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import librosa


# =========================================================
# 1. 回帰スプライン (regression_spline_csv) の読み込み
# =========================================================

@dataclass
class SplineModel:
    egg: str          # "Oq" or "Rq"
    audio: str        # "f0_hz" などの音響特徴量名
    model: object     # unpickled spline model
    val_mse: float
    x_min: float
    x_max: float


class RegressionBank:
    """
    regression_spline_csv に入っている Oq / Rq 用スプラインモデル群を管理するクラス。
    """

    def __init__(self, csv_path: str):
        df = pd.read_csv(csv_path)
        self.models: Dict[Tuple[str, str], SplineModel] = {}

        for _, row in df.iterrows():
            egg = str(row["egg"])          # "Oq" / "Rq"
            audio = str(row["audio"])      # "f0_hz" 等
            method = str(row.get("method", ""))
            mode = str(row.get("mode", ""))

            # 必要なら method/mode でフィルタ（とりあえず全部ロード）
            model_b64 = row["model_b64"]
            x_min = float(row.get("x_min", -np.inf))
            x_max = float(row.get("x_max", np.inf))
            val_mse = float(row.get("val_mse", 1.0))

            try:
                model_bytes = base64.b64decode(model_b64)
                model = pickle.loads(model_bytes)
            except Exception as e:
                print(f"[WARN] spline decode failed for {egg}/{audio}: {e}")
                continue

            key = (egg, audio)
            self.models[key] = SplineModel(
                egg=egg,
                audio=audio,
                model=model,
                val_mse=val_mse,
                x_min=x_min,
                x_max=x_max,
            )

        if not self.models:
            print("[WARN] No spline models were loaded from reg_spline_csv.")

    def predict_egg(self, egg: str, features: Dict[str, float]) -> Optional[float]:
        """
        音響特徴量 dict (audio_name -> scalar) から egg ("Oq"/"Rq") の推定値を返す。
        - 複数の audio 特徴量に対するモデルがあるので，
          それぞれのモデルの出力を 1/val_mse 重みで平均する。
        - feature が欠けている audio は無視。
        """
        preds = []
        weights = []

        for (egg_name, audio_name), sm in self.models.items():
            if egg_name != egg:
                continue
            if audio_name not in features:
                continue

            x = float(features[audio_name])
            # 安全のため範囲でクリップ
            x_clip = np.clip(x, sm.x_min, sm.x_max)

            try:
                y = float(sm.model(x_clip))
            except Exception:
                continue

            w = 1.0 / max(sm.val_mse, 1e-6)
            preds.append(y)
            weights.append(w)

        if not preds:
            return None

        preds = np.asarray(preds)
        weights = np.asarray(weights)
        return float(np.sum(preds * weights) / np.sum(weights))


# =========================================================
# 2. パーセンタイル / 統計 (param_percentiles, extended_stats)
# =========================================================

# ===========================================================
# GMM Parameter Loader
# ===========================================================
@dataclass
class GMMParams:
    # dict[(egg, audio)] = list of clusters
    bank: Dict[Tuple[str, str], List[Dict[str, float]]]

def load_gmm_params(csv_path: str) -> GMMParams:
    df = pd.read_csv(csv_path)
    bank = {}

    for _, row in df.iterrows():
        key = (row["egg"], row["audio"])
        if key not in bank:
            bank[key] = []

        bank[key].append({
            "weight": float(row["weight"]),
            "mu_y": float(row["mu_y"]),
            "cov_yy": float(row["cov_yy"]),
        })

    return GMMParams(bank=bank)

# ===========================================================
# GMM-based lightweight Gaussian loss
# ===========================================================
def gmm_loss(
    feature_value: float,
    egg: str,
    audio: str,
    gmm_bank: GMMParams,
) -> float:
    key = (egg, audio)
    if key not in gmm_bank.bank:
        return 0.0  # no GMM model → ignore

    L = 0.0
    for cl in gmm_bank.bank[key]:
        diff = feature_value - cl["mu_y"]
        var = cl["cov_yy"] if cl["cov_yy"] > 1e-8 else 1e-8
        L += cl["weight"] * (diff * diff / var)

    return float(L)



class PercentileBank:
    """
    param_percentiles / extended_stats から
    - 各パーセンタイル index(1〜5) に対応する Oq, Rq の値
    - Oq, Rq, vibrato_cents_abs の標準偏差 std
    を取り出すクラス。
    """

    def __init__(
        self,
        percentiles_csv: str,
        extended_csv: str,
        segment_id: str = "ALL",
        quality: str = "CLEAN",
    ):
        # -----------------------------
        # 初期化
        # -----------------------------
        # param_percentiles 用（Oq/Rq/vibrato の p12.5〜p87.5）
        self.segment_id = segment_id     # ← param_percentiles 用

        # extended_stats 用は "ALL" で固定
        # （あなたの CSV 仕様：extended_stats は ALL のみ）
        self.stats_segment_id = "ALL"

        # quality
        self.quality = quality

        # CSV 読み込み
        self.percentiles = pd.read_csv(percentiles_csv)
        self.extended = pd.read_csv(extended_csv)

        # パーセンタイル列
        self.idx_to_col = {
            1: "p12.5",
            2: "p25",
            3: "p50",
            4: "p75",
            5: "p87.5",
        }

    # -----------------------------------------------------
    # Oq/Rq など param_percentiles から値を取得
    # -----------------------------------------------------
    def get_target_value(self, param: str, idx: int) -> float:
        col = self.idx_to_col[idx]
        df = self.percentiles

        cond = (
            (df["segment_id"] == self.segment_id)  # ← param_percentiles は引数の segment_id を使う
            & (df["quality"] == self.quality)
            & (df["param"] == param)
        )

        sub = df.loc[cond, col]
        if sub.empty:
            raise ValueError(
                f"No percentile data for {param}, segment={self.segment_id}, quality={self.quality}"
            )

        return float(sub.iloc[0])

    # -----------------------------------------------------
    # extended_stats から std を読み出す
    # segment_id は常に stats_segment_id ("ALL")
    # -----------------------------------------------------
    def get_std(self, param: str) -> float:
        df = self.extended

        cond = (
            (df["segment_id"] == self.stats_segment_id)  # ← extended_stats は ALL 固定
            & (df["quality"] == self.quality)
            & (df["param"] == param)
        )

        sub = df.loc[cond, "std"]
        if sub.empty:
            raise ValueError(
                f"No std data for {param}, segment={self.stats_segment_id}, quality={self.quality}"
            )

        return float(sub.iloc[0])

    # -----------------------------------------------------
    # extended_stats から mean, std の dict を取得
    # -----------------------------------------------------
    def get_stats(self, param: str) -> dict:
        df = self.extended

        cond = (
            (df["segment_id"] == self.stats_segment_id)  # ← ALL 固定
            & (df["quality"] == self.quality)
            & (df["param"] == param)
        )

        sub = df.loc[cond]
        if sub.empty:
            raise ValueError(
                f"No extended stats for {param}, segment={self.stats_segment_id}, quality={self.quality}"
            )

        row = sub.iloc[0]
        return {
            "mean": float(row["mean"]),
            "std": float(row["std"]),
        }





# =========================================================
# 3. 音響特徴量抽出（簡易版）
# =========================================================

def extract_acoustic_features(
    wav_path: str,
    sr: int = 48000,
) -> Dict[str, float]:
    """
    GMM_additional に近い感じの静的音響特徴量を一括で推定。
    - f0_hz: YIN で推定した F0 の中央値
    - spec_centroid_hz: スペクトル重心の平均
    - spec_bandwidth_hz: スペクトル帯域幅の平均
    - zcr: ゼロ交差率（平均）
    - rms_dbfs: RMS を dBFS 化
    - mfcc1..mfcc5: 各 MFCC の平均
    - vibrato_cents_abs: F0 をセントにして絶対値の平均（簡易版）
    - cpp_snr: とりあえず 0.0 （必要なら差し替え）

    ※ WORLD aperiodicity は使用しない。
    """

    y, fs = librosa.load(wav_path, sr=sr)
    if y.ndim > 1:
        y = np.mean(y, axis=1)

    # --- F0 (YIN) ---
    f0 = librosa.yin(
        y,
        fmin=50.0,
        fmax=1000.0,
        sr=fs,
    )
    f0_valid = f0[f0 > 0]
    if len(f0_valid) == 0:
        f0_mean = 0.0
    else:
        f0_mean = float(np.median(f0_valid))

    # --- spectral features ---
    hop_length = 512
    centroid = librosa.feature.spectral_centroid(y=y, sr=fs, hop_length=hop_length)
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=fs, hop_length=hop_length)
    zcr = librosa.feature.zero_crossing_rate(y, hop_length=hop_length)
    rms = librosa.feature.rms(y=y, hop_length=hop_length)

    spec_centroid_hz = float(np.mean(centroid))
    spec_bandwidth_hz = float(np.mean(bandwidth))
    zcr_mean = float(np.mean(zcr))

    rms_mean = float(np.mean(rms))
    eps = 1e-12
    rms_dbfs = float(20.0 * np.log10(rms_mean + eps))

    # --- MFCC ---
    mfcc = librosa.feature.mfcc(y=y, sr=fs, n_mfcc=5, hop_length=hop_length)
    mfcc_means = [float(np.mean(mfcc[i])) for i in range(mfcc.shape[0])]

    # --- vibrato_cents_abs (簡易) ---
    if len(f0_valid) > 0:
        f0_med = np.median(f0_valid)
        cents = 1200.0 * np.log2(f0_valid / f0_med)
        vibrato_cents_abs = float(np.mean(np.abs(cents)))
    else:
        vibrato_cents_abs = 0.0

    # --- CPP (ダミー; 必要なら実装に差し替え) ---
    cpp_snr = 0.0

    feat: Dict[str, float] = {
        "f0_hz": f0_mean,
        "spec_centroid_hz": spec_centroid_hz,
        "spec_bandwidth_hz": spec_bandwidth_hz,
        "zcr": zcr_mean,
        "rms_dbfs": rms_dbfs,
        "vibrato_cents_abs": vibrato_cents_abs,
        "cpp_snr": cpp_snr,
    }
    for i, v in enumerate(mfcc_means, start=1):
        feat[f"mfcc{i}"] = v

    return feat


# =========================================================
# 4. Oq / Rq / vibrato の誤差定義 & 最尤インデックス選択
# =========================================================

def select_oq_rq_indices(
    features: Dict[str, float],
    reg_bank: RegressionBank,
    perc_bank: PercentileBank,
    gmm_bank: GMMParams,
    oq_candidates: List[int],
    rq_candidates: List[int],
) -> Tuple[int, int, Dict[str, float]]:
    """
    現在の音響特徴量から Oq/Rq の最尤パーセンタイルを決定する。
    スコアは L_total = L_spline + L_z_total + L_gmm_total。
    """

    # --- spline predicted values ---
    oq_hat = reg_bank.predict_egg("Oq", features)
    rq_hat = reg_bank.predict_egg("Rq", features)
    if oq_hat is None or rq_hat is None:
        raise RuntimeError("Failed to predict Oq or Rq. Check regression_spline_csv.")

    # --- std from extended_stats ---
    std_oq = perc_bank.get_std("Oq")
    std_rq = perc_bank.get_std("Rq")

    # --- debug store ---
    debug_info = {
        "Oq_hat": oq_hat,
        "Rq_hat": rq_hat,
        "std_Oq": std_oq,
        "std_Rq": std_rq,
        "candidates": [],
    }

    # ========== inner helper: L_z ==========
    def compute_L_z(pred_value: float, stats: dict) -> float:
        mu = float(stats["mean"])
        sd = float(stats["std"])
        if sd <= 1e-9:
            return (pred_value - mu)**2
        return ((pred_value - mu) / sd)**2

    # ========== inner helper: L_gmm ==========
    def compute_L_gmm(pred_value: float, gmm_list: List[Dict[str, float]]) -> float:
        if gmm_list is None:
            return 0.0
        total = 0.0
        for g in gmm_list:
            mu = float(g["mu_y"])
            var = float(g["cov_yy"]) if g["cov_yy"] > 1e-12 else 1e-12
            w = float(g["weight"])
            total += w * ((pred_value - mu) ** 2 / var)
        return total

    # --- best solution holder ---
    best_L = None
    best_oq = None
    best_rq = None

    # load stats
    stats_oq = perc_bank.get_stats("Oq")
    stats_rq = perc_bank.get_stats("Rq")

    # load GMM lists
    gmm_list_oq = gmm_bank.bank.get(("Oq", "f0_hz"), [])
    gmm_list_rq = gmm_bank.bank.get(("Rq", "f0_hz"), [])

    # =======================================================
    # search oq_idx × rq_idx
    # =======================================================
    for oq_idx in oq_candidates:
        t_oq = perc_bank.get_target_value("Oq", oq_idx)

        for rq_idx in rq_candidates:
            t_rq = perc_bank.get_target_value("Rq", rq_idx)

            # --- L_spline ---
            e_oq = (oq_hat - t_oq) / std_oq
            e_rq = (rq_hat - t_rq) / std_rq
            L_spline = e_oq**2 + e_rq**2

            # --- L_z ---
            L_z_oq = compute_L_z(oq_hat, stats_oq)
            L_z_rq = compute_L_z(rq_hat, stats_rq)
            L_z_total = L_z_oq + L_z_rq

            # --- L_gmm ---
            L_gmm_oq = compute_L_gmm(oq_hat, gmm_list_oq)
            L_gmm_rq = compute_L_gmm(rq_hat, gmm_list_rq)
            L_gmm_total = L_gmm_oq + L_gmm_rq

            # --- final L ---
            L = L_spline + L_z_total + L_gmm_total

            # store for debug
            debug_info["candidates"].append({
                "oq_idx": oq_idx,
                "rq_idx": rq_idx,
                "target_Oq": t_oq,
                "target_Rq": t_rq,
                "L_spline": float(L_spline),
                "L_z": float(L_z_total),
                "L_gmm": float(L_gmm_total),
                "L_total": float(L),
            })

            # update minimum
            if best_L is None or L < best_L:
                best_L = L
                best_oq = oq_idx
                best_rq = rq_idx

    if best_oq is None:
        raise RuntimeError("No valid Oq/Rq candidate found.")

    debug_info["best_L"] = float(best_L)
    debug_info["best_oq_idx"] = int(best_oq)
    debug_info["best_rq_idx"] = int(best_rq)

    return int(best_oq), int(best_rq), debug_info



def select_vibrato_index_from_percentiles(
    vibrato_cents_abs: float,
    perc_bank: PercentileBank,
) -> Tuple[int, Dict[str, float]]:

    vib_targets = {}
    for idx in range(1, 6):
        col = perc_bank.idx_to_col[idx]
        df = perc_bank.extended
        cond = (
            (df["segment_id"] == perc_bank.stats_segment_id)   # ★ 修正点：ALL を使う
            & (df["quality"] == perc_bank.quality)
            & (df["param"] == "vibrato_cents_abs")
        )
        sub = df.loc[cond, col]
        if sub.empty:
            raise ValueError("No vibrato_cents_abs percentile data.")
        vib_targets[idx] = float(sub.iloc[0])

    best_idx = None
    best_L = None

    debug = {
        "vibrato_cents_abs": float(vibrato_cents_abs),
        "candidates": []
    }

    for idx, target_v in vib_targets.items():
        L = (vibrato_cents_abs - target_v) ** 2
        debug["candidates"].append({
            "idx": idx,
            "target_v": float(target_v),
            "L": float(L)
        })
        if best_L is None or L < best_L:
            best_L = L
            best_idx = idx

    debug["best_vib_idx"] = int(best_idx)
    debug["best_L"] = float(best_L)
    return int(best_idx), debug




# =========================================================
# 5. ファイル名の組み立て
# =========================================================

def build_audio_filename(
    base_prefix: str,
    oq_idx: int,
    rq_idx: int,
    vib_idx: int,
    segment_index: Optional[int] = None,
    ext: str = ".wav",
) -> str:
    """
    Concone ファイル名のパターン例:
        concone_synthesizerV_SakiAI_1_1_1_seg10.wav

    segment_index が None の場合:
        concone_synthesizerV_SakiAI_1_1_1.wav
    """
    if segment_index is None:
        return f"{base_prefix}_{oq_idx}_{rq_idx}_{vib_idx}{ext}"
    else:
        return f"{base_prefix}_{oq_idx}_{rq_idx}_{vib_idx}_seg{segment_index:02d}{ext}"


# =========================================================
# 6. メイン (オフライン実行用のラッパ)
# =========================================================

def main():
    parser = argparse.ArgumentParser(
        description="Concone real-time-like file selector core (Oq/Rq/vibrato)."
    )

    # 入力 wav（オフラインテスト用）
    parser.add_argument(
        "--wav",
        type=str,
        required=True,
        help="テスト用の入力 wav ファイル（ユーザ歌唱）",
    )

    # 学習済み CSV 群
    parser.add_argument(
        "--percentiles_csv",
        type=str,
        required=True,
        help="param_percentiles CSV のパス",
    )
    parser.add_argument(
        "--extended_csv",
        type=str,
        required=True,
        help="extended_stats CSV のパス",
    )
    parser.add_argument(
        "--reg_spline_csv",
        type=str,
        required=True,
        help="egg_vs_audio_regressions_spline CSV のパス",
    )

    parser.add_argument(
        "--segment_id",
        type=str,
        default="ALL",
        help="param_percentiles / extended_stats で使う segment_id (例: seg10, ALL)",
    )
    parser.add_argument(
        "--quality",
        type=str,
        default="CLEAN",
        help="param_percentiles / extended_stats で使う quality (通常 CLEAN)",
    )

    # Oq / Rq / vibrato の候補
    parser.add_argument(
        "--oq_candidates",
        type=int,
        nargs="+",
        default=[1, 2, 3, 4, 5],
        help="Oq パーセンタイル候補 (1〜5の整数)",
    )
    parser.add_argument(
        "--rq_candidates",
        type=int,
        nargs="+",
        default=[1, 2, 3, 4, 5],
        help="Rq パーセンタイル候補 (1〜5の整数)",
    )

    # 音声ファイル名のプレフィックス / seg 番号
    parser.add_argument(
        "--base_prefix",
        type=str,
        required=True,
        help="例: concone_synthesizerV_SakiAI",
    )
    parser.add_argument(
        "--segment_index",
        type=int,
        default=None,
        help="segNN の NN 部分。None の場合は seg を付けない。",
    )

    parser.add_argument(
    "--gmm_csv",
    type=str,
    required=True,
    help="gmm_params CSV のパス",
)


    # 出力 JSON
    parser.add_argument(
        "--out_json",
        type=str,
        default=None,
        help="推定結果を JSON として保存するパス (任意)",
    )

    args = parser.parse_args()

    # 1) 特徴量抽出
    print(f"[INFO] Extracting features from {args.wav}")
    feat = extract_acoustic_features(args.wav)

    # 2) モデル / パーセンタイル ロード
    reg_bank = RegressionBank(args.reg_spline_csv)
    perc_bank = PercentileBank(
        percentiles_csv=args.percentiles_csv,
        extended_csv=args.extended_csv,
        segment_id=args.segment_id,
        quality=args.quality,
    )

    # 3) Oq / Rq index
    gmm_bank = load_gmm_params(args.gmm_csv)

    oq_idx, rq_idx, debug_oqrq = select_oq_rq_indices(
        features=feat,
        reg_bank=reg_bank,
        perc_bank=perc_bank,
        gmm_bank=gmm_bank,
        oq_candidates=args.oq_candidates,
        rq_candidates=args.rq_candidates,
    )


    # 4) vibrato index
    vib_idx, debug_vib = select_vibrato_index_from_percentiles(
        vibrato_cents_abs=feat["vibrato_cents_abs"],
        perc_bank=perc_bank,
    )


    # 5) ファイル名
    fname = build_audio_filename(
        base_prefix=args.base_prefix,
        oq_idx=oq_idx,
        rq_idx=rq_idx,
        vib_idx=vib_idx,
        segment_index=args.segment_index,
        ext=".wav",
    )

    result = {
        "wav_input": args.wav,
        "features": feat,
        "OqRq_debug": debug_oqrq,
        "vib_debug": debug_vib,
        "best_oq_idx": oq_idx,
        "best_rq_idx": rq_idx,
        "best_vib_idx": vib_idx,
        "selected_filename": fname,
    }

    print("===== RESULT =====")
    print(json.dumps(result, ensure_ascii=False, indent=2))

    if args.out_json is not None:
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"[SAVE] {args.out_json}")


if __name__ == "__main__":
    main()
