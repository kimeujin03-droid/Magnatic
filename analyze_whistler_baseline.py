import json
import os
from pathlib import Path

import cdflib
import numpy as np
import pandas as pd

from analyze_whistler_burst import DATA_DIR, FGM_BRST_FILES, FGM_SRVY_FILE, SCM_BURST_FILES, load_epoch


BASE_DIR = Path(os.environ.get("MMS_CASE_DIR", r"C:\Magnetic"))
BASELINE_DIR = BASE_DIR / "baseline_santolik"
CONFIG_PATH = BASELINE_DIR / "baseline_config.json"
SEGMENTS_CSV = BASELINE_DIR / "baseline_segments.csv"
EVENTS_CSV = BASELINE_DIR / "baseline_events.csv"
SUMMARY_PATH = BASELINE_DIR / "baseline_summary.md"
STATUS_PATH = BASELINE_DIR / "baseline_status.json"

STFT_N = 4096
STFT_STEP = 2048
FREQ_SMOOTH_BINS = 1
MIN_EVENT_DURATION_S = 0.25
MAX_EVENT_GAP_S = 1.0


def load_config() -> dict:
    with open(CONFIG_PATH, "r", encoding="utf-8") as fh:
        return json.load(fh)


def load_fgm() -> pd.DataFrame:
    frames = []
    for path in FGM_BRST_FILES:
        try:
            cdf = cdflib.CDF(path)
            t = load_epoch(cdf)
            vars = cdf.cdf_info().zVariables
            b_var = next(v for v in vars if v.startswith("mms1_fgm_b_gse_"))
            b = np.asarray(cdf.varget(b_var), dtype=np.float64)
            frames.append(pd.DataFrame({"time": t, "Bt": b[:, 3]}))
        except Exception:
            continue
    if frames:
        return pd.concat(frames, ignore_index=True).sort_values("time").set_index("time")

    cdf = cdflib.CDF(FGM_SRVY_FILE)
    t = load_epoch(cdf)
    vars = cdf.cdf_info().zVariables
    b_var = next(v for v in vars if v.startswith("mms1_fgm_b_gse_"))
    b = np.asarray(cdf.varget(b_var), dtype=np.float64)
    return pd.DataFrame({"time": t, "Bt": b[:, 3]}).set_index("time")


def load_fgm_interpolator() -> tuple[np.ndarray, np.ndarray]:
    fgm = load_fgm()
    return np.asarray(fgm.index.view("int64")), fgm["Bt"].to_numpy(dtype=np.float64)


def smoothed_spectral_matrix(fft_xyz: np.ndarray, fs: float, window: np.ndarray) -> np.ndarray:
    scale = fs * np.sum(window**2)
    outer = fft_xyz[:, :, None] * np.conjugate(fft_xyz[:, None, :])
    outer = outer / scale
    if len(outer) > 2:
        outer[1:-1] *= 2.0

    csum = np.cumsum(outer, axis=0)
    padded = np.concatenate([np.zeros((1, 3, 3), dtype=np.complex128), csum], axis=0)
    idx = np.arange(len(outer))
    lo = np.clip(idx - FREQ_SMOOTH_BINS, 0, len(outer))
    hi = np.clip(idx + FREQ_SMOOTH_BINS + 1, 0, len(outer))
    counts = (hi - lo).reshape(-1, 1, 1)
    summed = padded[hi] - padded[lo]
    return summed / np.maximum(counts, 1)


def planarity_from_eigs(evals_desc: np.ndarray) -> float:
    lam1, lam2, lam3 = evals_desc
    denom = lam1 + lam2 + lam3
    if denom <= 0:
        return np.nan
    return float(1.0 - (2.0 * lam3 / max(lam1 + lam2, 1e-12)))


def ellipticity_from_vector(polarization_vec: np.ndarray, k_hat: np.ndarray) -> float:
    proj = polarization_vec - np.vdot(k_hat, polarization_vec) * k_hat
    u = np.real(proj)
    v = np.imag(proj)
    uu = float(np.dot(u, u))
    vv = float(np.dot(v, v))
    uv = float(np.dot(u, v))
    discr = max((uu - vv) ** 2 + 4.0 * (uv**2), 0.0)
    major2 = 0.5 * (uu + vv + np.sqrt(discr))
    minor2 = 0.5 * (uu + vv - np.sqrt(discr))
    if major2 <= 0:
        return np.nan
    handedness = np.sign(np.dot(k_hat.real, np.cross(u, v)))
    return float(handedness * np.sqrt(max(minor2, 0.0) / major2))


def segment_rows(path: str, fgm_x: np.ndarray, fgm_bt: np.ndarray, cfg: dict) -> list[dict]:
    cdf = cdflib.CDF(path)
    t = load_epoch(cdf)
    data = np.asarray(cdf.varget("mms1_scm_acb_gse_schb_brst_l2"), dtype=np.float64)
    dt = float(np.median(t.to_series().diff().dropna().dt.total_seconds()))
    fs = 1.0 / dt
    window = np.hanning(STFT_N)

    f_low = cfg["frequency_gate"]["low_fraction_of_fce"]
    f_high = cfg["frequency_gate"]["high_fraction_of_fce"]
    ell_min = cfg["quality_gate"]["ellipticity_min"]
    plan_min = cfg["quality_gate"]["planarity_min"]
    psd_min = cfg["quality_gate"]["magnetic_psd_min_nt2_per_hz"]

    rows = []
    for start in range(0, len(data) - STFT_N + 1, STFT_STEP):
        seg = data[start : start + STFT_N]
        center_t = t[start + STFT_N // 2]
        center_ns = np.int64(center_t.value)
        bt = float(np.interp(center_ns, fgm_x, fgm_bt))
        fce_hz = 28.0 * bt
        fft_xyz = np.fft.rfft(seg * window[:, None], axis=0)
        freqs = np.fft.rfftfreq(STFT_N, d=dt)
        sm = smoothed_spectral_matrix(fft_xyz, fs, window)

        low = f_low * fce_hz
        high = f_high * fce_hz
        band_idx = np.where((freqs >= low) & (freqs <= high))[0]

        best = None
        if len(band_idx) == 0:
            continue

        band_sm = sm[band_idx]
        finite_mask = np.isfinite(np.real(band_sm)).all(axis=(1, 2)) & np.isfinite(np.imag(band_sm)).all(axis=(1, 2))
        if not finite_mask.any():
            continue
        band_idx = band_idx[finite_mask]
        band_sm = band_sm[finite_mask]

        try:
            evals_all, evecs_all = np.linalg.eigh(band_sm)
        except np.linalg.LinAlgError:
            continue

        for local_i, idx in enumerate(band_idx):
            evals = evals_all[local_i]
            evecs = evecs_all[local_i]
            order = np.argsort(evals)[::-1]
            evals_desc = np.real(evals[order])
            evecs_desc = evecs[:, order]
            planarity = planarity_from_eigs(evals_desc)
            ellipticity = ellipticity_from_vector(evecs_desc[:, 0], evecs_desc[:, -1])
            psd = float(np.real(np.trace(band_sm[local_i])))
            passes = (
                freqs[idx] >= low
                and freqs[idx] <= high
                and ellipticity == ellipticity
                and planarity == planarity
                and ellipticity > ell_min
                and planarity > plan_min
                and psd > psd_min
            )
            row = {
                "time": center_t,
                "burst_file": os.path.basename(path),
                "fce_hz": fce_hz,
                "peak_freq_hz": float(freqs[idx]),
                "freq_fraction_of_fce": float(freqs[idx] / fce_hz) if fce_hz > 0 else np.nan,
                "ellipticity": float(ellipticity) if ellipticity == ellipticity else np.nan,
                "planarity": float(planarity) if planarity == planarity else np.nan,
                "psd_nt2_per_hz": psd,
                "baseline_pass": bool(passes),
            }
            if best is None or row["psd_nt2_per_hz"] > best["psd_nt2_per_hz"]:
                best = row
        if best is not None:
            rows.append(best)
    return rows


def build_segments(cfg: dict) -> pd.DataFrame:
    fgm_x, fgm_bt = load_fgm_interpolator()
    rows = []
    for path in SCM_BURST_FILES:
        rows.extend(segment_rows(path, fgm_x, fgm_bt, cfg))
    df = pd.DataFrame(rows).sort_values("time").reset_index(drop=True)
    df.to_csv(SEGMENTS_CSV, index=False)
    return df


def build_events(segments: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    valid = segments[segments["baseline_pass"]].copy()
    if valid.empty:
        out = pd.DataFrame(
            columns=[
                "baseline_event_id",
                "start_time",
                "end_time",
                "duration_s",
                "peak_time",
                "peak_freq_hz",
                "mean_freq_fraction_of_fce",
                "mean_ellipticity",
                "mean_planarity",
                "peak_psd_nt2_per_hz",
                "num_segments",
            ]
        )
        out.to_csv(EVENTS_CSV, index=False)
        return out

    times = pd.to_datetime(valid["time"])
    group = (times.diff() > pd.Timedelta(seconds=cfg["event_merge"]["maximum_gap_s"])).cumsum()
    rows = []
    step_s = STFT_STEP / 16384.0
    next_event_id = 1
    for _, chunk in valid.groupby(group):
        start_time = pd.Timestamp(chunk["time"].min())
        end_time = pd.Timestamp(chunk["time"].max())
        duration_s = (end_time - start_time).total_seconds() + step_s
        if duration_s < cfg["event_merge"]["minimum_duration_s"]:
            continue
        peak_idx = chunk["psd_nt2_per_hz"].idxmax()
        peak = chunk.loc[peak_idx]
        rows.append(
            {
                "baseline_event_id": next_event_id,
                "start_time": start_time,
                "end_time": end_time,
                "duration_s": duration_s,
                "peak_time": peak["time"],
                "peak_freq_hz": float(peak["peak_freq_hz"]),
                "mean_freq_fraction_of_fce": float(chunk["freq_fraction_of_fce"].mean()),
                "mean_ellipticity": float(chunk["ellipticity"].mean()),
                "mean_planarity": float(chunk["planarity"].mean()),
                "peak_psd_nt2_per_hz": float(chunk["psd_nt2_per_hz"].max()),
                "num_segments": int(len(chunk)),
            }
        )
        next_event_id += 1
    out = pd.DataFrame(rows)
    out.to_csv(EVENTS_CSV, index=False)
    return out


def write_summary(cfg: dict, segments: pd.DataFrame, events: pd.DataFrame) -> None:
    current_events_path = BASE_DIR / "whistler_events.csv"
    current_event_count = 0
    if current_events_path.exists():
        current_event_count = len(pd.read_csv(current_events_path))

    passing = segments[segments["baseline_pass"]].copy()
    lines = [
        "# Baseline Whistler Summary",
        "",
        "## Rule",
        f"- Frequency gate: `{cfg['frequency_gate']['low_fraction_of_fce']:.1f} fce` to `{cfg['frequency_gate']['high_fraction_of_fce']:.1f} fce`",
        f"- Ellipticity > `{cfg['quality_gate']['ellipticity_min']:.1f}`",
        f"- Planarity > `{cfg['quality_gate']['planarity_min']:.1f}`",
        f"- PSD > `{cfg['quality_gate']['magnetic_psd_min_nt2_per_hz']}` nT^2/Hz",
        "",
        "## Counts",
        f"- Total evaluated segments: `{len(segments)}`",
        f"- Passing baseline segments: `{int(segments['baseline_pass'].sum()) if not segments.empty else 0}`",
        f"- Baseline events: `{len(events)}`",
        f"- Current detector events: `{current_event_count}`",
        "",
    ]

    if not passing.empty:
        lines.extend(
            [
                "## Passing Segment Summary",
                f"- Median f/fce: `{passing['freq_fraction_of_fce'].median():.3f}`",
                f"- Median ellipticity: `{passing['ellipticity'].median():.3f}`",
                f"- Median planarity: `{passing['planarity'].median():.3f}`",
                f"- Median PSD: `{passing['psd_nt2_per_hz'].median():.3e}` nT^2/Hz",
                "",
                "## Top Baseline Events",
                "| Event | Start | End | Duration [s] | Peak Freq [Hz] | Mean f/fce | Mean Ellipticity | Mean Planarity | Peak PSD |",
                "|---|---|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        top = events.sort_values("peak_psd_nt2_per_hz", ascending=False).head(10)
        for _, row in top.iterrows():
            lines.append(
                f"| {int(row['baseline_event_id'])} | {row['start_time']} | {row['end_time']} | "
                f"{row['duration_s']:.3f} | {row['peak_freq_hz']:.2f} | {row['mean_freq_fraction_of_fce']:.3f} | "
                f"{row['mean_ellipticity']:.3f} | {row['mean_planarity']:.3f} | {row['peak_psd_nt2_per_hz']:.3e} |"
            )
    else:
        lines.extend(["## Result", "- No segments satisfied the baseline Santolik-style rule."])

    lines.extend(
        [
            "",
            f"Segments CSV: `{SEGMENTS_CSV}`",
            f"Events CSV: `{EVENTS_CSV}`",
        ]
    )

    with open(SUMMARY_PATH, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))


def write_status(cfg: dict, segments: pd.DataFrame, events: pd.DataFrame) -> None:
    status = {
        "ready": True,
        "implemented": True,
        "detector": "baseline_santolik_style",
        "config": cfg,
        "data_dir": str(DATA_DIR),
        "segment_count": int(len(segments)),
        "passing_segment_count": int(segments["baseline_pass"].sum()) if not segments.empty else 0,
        "event_count": int(len(events)),
    }
    with open(STATUS_PATH, "w", encoding="utf-8") as fh:
        json.dump(status, fh, indent=2, default=str)


def main() -> None:
    BASELINE_DIR.mkdir(parents=True, exist_ok=True)
    cfg = load_config()
    segments = build_segments(cfg)
    events = build_events(segments, cfg)
    write_summary(cfg, segments, events)
    write_status(cfg, segments, events)
    print(f"baseline workspace: {BASELINE_DIR}")
    print(f"segments: {SEGMENTS_CSV}")
    print(f"events: {EVENTS_CSV}")
    print(f"summary: {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
