import glob
import os

import cdflib
import numpy as np
import pandas as pd


BASE_DIR = os.environ.get("MMS_CASE_DIR", r"C:\Magnetic")
DATA_DIR = os.path.join(BASE_DIR, "data") if os.path.isdir(os.path.join(BASE_DIR, "data")) else BASE_DIR
FGM_SRVY_FILE = sorted(glob.glob(os.path.join(DATA_DIR, "mms1_fgm_srvy_l2_*.cdf")))[0]
FGM_BRST_FILES = sorted(glob.glob(os.path.join(DATA_DIR, "mms1_fgm_brst_l2_*.cdf")))
FPI_FILES = sorted(glob.glob(os.path.join(DATA_DIR, "mms1_fpi_brst_l2_dis-moms_*.cdf")))
SCM_BURST_FILES = sorted(glob.glob(os.path.join(DATA_DIR, "mms1_scm_brst_l2_schb_*.cdf")))

REPORT_PATH = os.path.join(BASE_DIR, "whistler_burst_summary.md")
CSV_PATH = os.path.join(BASE_DIR, "whistler_burst_candidates.csv")
SVG_PATH = os.path.join(BASE_DIR, "whistler_burst_overview.svg")

BIN_SECONDS = 5
VX_THRESHOLD = 300.0
MIN_CONSECUTIVE_BINS = 2
BZ_VAR_THRESHOLD = 3.0

STFT_N = 4096
STFT_STEP = 2048
WHISTLER_LOW_FRACTION = 0.05
WHISTLER_HIGH_FRACTION = 0.5
MIN_FCE_HZ = 50.0
BACKGROUND_WINDOW_SEGMENTS = 17
BACKGROUND_MIN_SEGMENTS = 5
BACKGROUND_QUANTILE = 0.2
MIN_BACKGROUND_EXCESS = 3.0


def load_epoch(cdf: cdflib.CDF) -> pd.DatetimeIndex:
    return pd.to_datetime(cdflib.cdfepoch.to_datetime(cdf.varget("Epoch")))


def robust_zscore(series: pd.Series) -> pd.Series:
    median = series.median()
    mad = (series - median).abs().median()
    scale = 1.4826 * mad if mad and not np.isnan(mad) else series.std()
    return (series - median) / scale


def consecutive_true(mask: pd.Series, min_run: int) -> pd.Series:
    group_id = mask.ne(mask.shift()).cumsum()
    run_lengths = mask.groupby(group_id).transform("sum")
    return mask & (run_lengths >= min_run)


def classify_vx_direction(vx: float) -> str:
    if vx >= VX_THRESHOLD:
        return "earthward_candidate"
    if vx <= -VX_THRESHOLD:
        return "tailward_fastflow_candidate"
    return "none"


def load_fpi() -> pd.DataFrame:
    frames = []
    for path in FPI_FILES:
        cdf = cdflib.CDF(path)
        t = load_epoch(cdf)
        vars = cdf.cdf_info().zVariables
        bulk_var = next(v for v in vars if "mms1_dis_bulkv_gse_" in v and ("_brst" in v or "_fast" in v))
        v = np.asarray(cdf.varget(bulk_var), dtype=np.float64)
        frames.append(pd.DataFrame({"time": t, "Vx": v[:, 0], "Vy": v[:, 1], "Vz": v[:, 2]}))
    return pd.concat(frames, ignore_index=True).sort_values("time").set_index("time")


def load_fgm() -> pd.DataFrame:
    frames = []
    for path in FGM_BRST_FILES:
        try:
            cdf = cdflib.CDF(path)
            t = load_epoch(cdf)
            vars = cdf.cdf_info().zVariables
            b_var = next(v for v in vars if v.startswith("mms1_fgm_b_gse_"))
            b = np.asarray(cdf.varget(b_var), dtype=np.float64)
            frames.append(pd.DataFrame({"time": t, "Bx": b[:, 0], "By": b[:, 1], "Bz": b[:, 2], "Bt": b[:, 3]}))
        except Exception:
            continue
    if frames:
        return pd.concat(frames, ignore_index=True).sort_values("time").set_index("time")

    cdf = cdflib.CDF(FGM_SRVY_FILE)
    t = load_epoch(cdf)
    vars = cdf.cdf_info().zVariables
    b_var = next(v for v in vars if v.startswith("mms1_fgm_b_gse_"))
    b = np.asarray(cdf.varget(b_var), dtype=np.float64)
    return pd.DataFrame({"time": t, "Bx": b[:, 0], "By": b[:, 1], "Bz": b[:, 2], "Bt": b[:, 3]}).set_index("time")


def build_bbf_table() -> pd.DataFrame:
    fpi = load_fpi().resample(f"{BIN_SECONDS}s").mean()
    fgm = load_fgm()[["Bz", "Bt"]].resample(f"{BIN_SECONDS}s").mean()
    merged = fpi.join(fgm, how="inner")
    merged["abs_vx"] = merged["Vx"].abs()
    merged["earthward_speed_flag"] = merged["Vx"] > VX_THRESHOLD
    merged["tailward_speed_flag"] = merged["Vx"] < -VX_THRESHOLD
    merged["bbf_speed_flag"] = merged["earthward_speed_flag"]
    merged["bbf_speed_run_flag"] = consecutive_true(merged["earthward_speed_flag"], MIN_CONSECUTIVE_BINS)
    merged["tailward_speed_run_flag"] = consecutive_true(merged["tailward_speed_flag"], MIN_CONSECUTIVE_BINS)
    merged["dVx"] = merged["Vx"].diff()
    merged["dBz"] = merged["Bz"].diff()
    merged["bz_delta"] = merged["Bz"].diff().abs()
    merged["bz_support_flag"] = merged["bz_delta"] > BZ_VAR_THRESHOLD
    merged["bbf_operational_flag"] = merged["bbf_speed_run_flag"] & merged["bz_support_flag"].fillna(False)
    merged["bbf_direction"] = merged["Vx"].apply(classify_vx_direction)
    return merged


def load_fgm_interpolator() -> tuple[np.ndarray, np.ndarray]:
    fgm = load_fgm()
    x = np.asarray(fgm.index.view("int64"))
    y = fgm["Bt"].to_numpy(dtype=np.float64)
    return x, y


def bandpower(spectrum_power: np.ndarray, freqs: np.ndarray, low_hz: float, high_hz: float) -> float:
    mask = (freqs >= low_hz) & (freqs < high_hz)
    return float(spectrum_power[mask].sum()) if mask.any() else float("nan")


def analyze_single_burst(path: str, fgm_x: np.ndarray, fgm_bt: np.ndarray) -> pd.DataFrame:
    cdf = cdflib.CDF(path)
    t = load_epoch(cdf)
    data = np.asarray(cdf.varget("mms1_scm_acb_gse_schb_brst_l2"), dtype=np.float64)
    dt = float(np.median(t.to_series().diff().dropna().dt.total_seconds()))
    fs = 1.0 / dt
    freqs = np.fft.rfftfreq(STFT_N, d=dt)
    window = np.hanning(STFT_N)

    times = []
    fce_list = []
    whistler_power = []
    total_power = []
    peak_freq = []

    for start in range(0, len(data) - STFT_N + 1, STFT_STEP):
        seg = data[start : start + STFT_N]
        center_t = t[start + STFT_N // 2]
        center_ns = np.int64(center_t.value)
        bt = float(np.interp(center_ns, fgm_x, fgm_bt))
        fce = 28.0 * bt

        px = np.abs(np.fft.rfft(seg[:, 0] * window)) ** 2
        py = np.abs(np.fft.rfft(seg[:, 1] * window)) ** 2
        pz = np.abs(np.fft.rfft(seg[:, 2] * window)) ** 2
        p = px + py + pz

        low = WHISTLER_LOW_FRACTION * fce
        high = WHISTLER_HIGH_FRACTION * fce
        times.append(center_t)
        fce_list.append(fce)
        whistler_power.append(bandpower(p, freqs, low, high))
        total_power.append(bandpower(p, freqs, 10.0, min(4000.0, freqs[-1])))
        band_mask = (freqs >= low) & (freqs < high)
        if band_mask.any():
            pf = float(freqs[band_mask][np.argmax(p[band_mask])])
        else:
            pf = float("nan")
        peak_freq.append(pf)

    df = pd.DataFrame(
        {
            "time": pd.DatetimeIndex(times),
            "fce_hz": fce_list,
            "whistler_band_low_hz": WHISTLER_LOW_FRACTION * np.asarray(fce_list),
            "whistler_band_high_hz": WHISTLER_HIGH_FRACTION * np.asarray(fce_list),
            "whistler_band_power": whistler_power,
            "wave_total_power_10_4000hz": total_power,
            "whistler_peak_freq_hz": peak_freq,
            "burst_file": os.path.basename(path),
        }
    ).set_index("time")
    df["whistler_ratio"] = df["whistler_band_power"] / df["wave_total_power_10_4000hz"]
    background = df["whistler_band_power"].rolling(
        window=BACKGROUND_WINDOW_SEGMENTS,
        center=True,
        min_periods=BACKGROUND_MIN_SEGMENTS,
    ).quantile(BACKGROUND_QUANTILE)
    global_background = df["whistler_band_power"].quantile(BACKGROUND_QUANTILE)
    df["whistler_background_power"] = background.fillna(global_background).clip(lower=1.0)
    df["background_excess"] = df["whistler_band_power"] / df["whistler_background_power"]
    df["fce_valid"] = df["fce_hz"] >= MIN_FCE_HZ
    return df


def build_whistler_table() -> pd.DataFrame:
    fgm_x, fgm_bt = load_fgm_interpolator()
    frames = [analyze_single_burst(path, fgm_x, fgm_bt) for path in SCM_BURST_FILES]
    wh = pd.concat(frames).sort_index()
    valid = wh["fce_valid"] & wh["whistler_ratio"].notna() & wh["background_excess"].notna()
    wh["whistler_power_z"] = np.nan
    wh["whistler_ratio_z"] = np.nan
    wh["background_excess_z"] = np.nan
    wh["whistler_score"] = np.nan
    if valid.any():
        wh.loc[valid, "whistler_power_z"] = robust_zscore(np.log10(wh.loc[valid, "whistler_band_power"].clip(lower=1.0)))
        wh.loc[valid, "whistler_ratio_z"] = robust_zscore(wh.loc[valid, "whistler_ratio"].clip(lower=0.0))
        wh.loc[valid, "background_excess_z"] = robust_zscore(np.log10(wh.loc[valid, "background_excess"].clip(lower=1.0)))
        wh.loc[valid, "whistler_score"] = (
            wh.loc[valid, "whistler_power_z"].clip(lower=0.0)
            + wh.loc[valid, "whistler_ratio_z"].clip(lower=0.0)
            + wh.loc[valid, "background_excess_z"].clip(lower=0.0)
        )
    return wh


def scale_points(values: np.ndarray, width: int, height: int, pad_x: int, pad_y: int, top: int) -> str:
    vmin = float(np.nanmin(values))
    vmax = float(np.nanmax(values))
    if vmax == vmin:
        vmax = vmin + 1.0
    x = np.linspace(pad_x, width - pad_x, num=len(values))
    y = top + pad_y + (height - 2 * pad_y) * (1.0 - (values - vmin) / (vmax - vmin))
    return " ".join(f"{xi:.1f},{yi:.1f}" for xi, yi in zip(x, y))


def render_svg(window: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> None:
    width = 1400
    panel_h = 140
    height = 620
    pad_x = 80
    pad_y = 22
    panels = [
        ("Vx", "Velocity Vx [km/s]", "#0b6e4f"),
        ("Bz", "Magnetic Field Bz [nT]", "#b22222"),
        ("whistler_band_power", "Whistler-Band Power [0.1-0.5 fce]", "#1f4e79"),
        ("whistler_score", "Whistler Score", "#6b4f9d"),
    ]
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{pad_x}" y="28" font-size="22" font-family="Arial">Whistler burst overview</text>',
        f'<text x="{pad_x}" y="50" font-size="14" font-family="Arial">{start} to {end}</text>',
    ]
    for idx, (col, title, color) in enumerate(panels):
        top = 70 + idx * (panel_h + 8)
        bottom = top + panel_h
        vals = window[col].to_numpy(dtype=float)
        parts.append(f'<rect x="{pad_x}" y="{top}" width="{width - 2 * pad_x}" height="{panel_h}" fill="none" stroke="#cccccc"/>')
        parts.append(f'<text x="{pad_x}" y="{top - 8}" font-size="16" font-family="Arial">{title}</text>')
        parts.append(f'<polyline fill="none" stroke="{color}" stroke-width="2" points="{scale_points(vals, width, panel_h, pad_x, pad_y, top)}"/>')
        parts.append(f'<text x="10" y="{top + 16}" font-size="12" font-family="Arial">{np.nanmax(vals):.2f}</text>')
        parts.append(f'<text x="10" y="{bottom - 6}" font-size="12" font-family="Arial">{np.nanmin(vals):.2f}</text>')
    parts.append("</svg>")
    with open(SVG_PATH, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))


def main() -> None:
    bbf = build_bbf_table()
    wh = build_whistler_table()

    bbf_for_join = bbf[["Vx", "Bz", "bbf_operational_flag", "bbf_direction"]].copy().reset_index()
    bbf_for_join = bbf_for_join.sort_values("time")
    joined = wh.copy()
    joined = joined.reset_index().sort_values("time")
    joined = pd.merge_asof(
        joined,
        bbf_for_join,
        on="time",
        direction="nearest",
        tolerance=pd.Timedelta(seconds=BIN_SECONDS / 2),
    ).set_index("time").sort_index()
    joined["bbf_operational_flag"] = joined["bbf_operational_flag"].fillna(False)
    joined["bbf_direction"] = joined["bbf_direction"].fillna("none")

    top = joined.sort_values("whistler_score", ascending=False).head(20).reset_index()
    top.to_csv(CSV_PATH, index=False)

    primary_time = joined["whistler_score"].idxmax()
    window_start = primary_time - pd.Timedelta(seconds=20)
    window_end = primary_time + pd.Timedelta(seconds=40)
    render_svg(joined.loc[window_start:window_end, ["Vx", "Bz", "whistler_band_power", "whistler_score"]], window_start, window_end)

    overlap = joined[joined["bbf_operational_flag"]]
    lines = [
        "# Whistler Burst Summary",
        "",
        "## Feasibility",
        f"- SCM burst sample rate is about `{1.0 / np.median(pd.to_datetime(cdflib.cdfepoch.to_datetime(cdflib.CDF(SCM_BURST_FILES[0]).varget('Epoch'))).to_series().diff().dropna().dt.total_seconds()):.2f} Hz`.",
        "- This is sufficient for direct whistler-band analysis in the current interval.",
        "- Operational whistler band was defined dynamically as `0.1-0.5 fce`, with `fce ≈ 28 * Bt[nT]` from FGM.",
        "",
        "## BBF Coupling",
        f"- Burst files analyzed: `{len(SCM_BURST_FILES)}`",
        f"- Whistler segments analyzed: `{len(joined)}`",
        f"- Segments inside provisional BBF bins: `{int(joined['bbf_operational_flag'].sum())}`",
        f"- Strongest whistler segment time: `{primary_time}`",
        "",
        "## Top Whistler Candidates",
        "| Time | Burst File | fce [Hz] | Whistler Band [Hz] | Peak Freq [Hz] | Whistler Power | Ratio | Score | BBF Flag | Direction |",
        "|---|---|---:|---|---:|---:|---:|---:|---|---|",
    ]

    for _, row in top.head(10).iterrows():
        lines.append(
            f"| {row['time']} | {row['burst_file']} | {row['fce_hz']:.1f} | "
            f"{row['whistler_band_low_hz']:.1f}-{row['whistler_band_high_hz']:.1f} | {row['whistler_peak_freq_hz']:.1f} | "
            f"{row['whistler_band_power']:.3f} | {row['whistler_ratio']:.4f} | {row['whistler_score']:.3f} | "
            f"{bool(row['bbf_operational_flag'])} | {row['bbf_direction']} |"
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "- `whistler_score` is a candidate-generation score based on dynamic whistler-band power and its fraction of total burst-band power.",
            "- It is not a final event label.",
            "- `bbf_operational_flag` remains the provisional BBF rule from the previous step.",
            "",
            f"Candidate table: `{CSV_PATH}`",
            f"Overview plot: `{SVG_PATH}`",
        ]
    )

    with open(REPORT_PATH, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))

    print(f"report: {REPORT_PATH}")
    print(f"plot: {SVG_PATH}")
    print(f"candidate csv: {CSV_PATH}")


if __name__ == "__main__":
    main()
