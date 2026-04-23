from dataclasses import dataclass

import cdflib
import numpy as np
import pandas as pd

from path_utils import repo_base_dir


BASE_DIR = repo_base_dir()
FGM_FILE = BASE_DIR / "mms1_fgm_srvy_l2_20260113_v5.540.0.cdf"
SCM_FILE = BASE_DIR / "mms1_scm_srvy_l2_scsrvy_20260113_v2.2.0.cdf"
FPI_FILES = [
    BASE_DIR / "mms1_fpi_fast_l2_dis-moms_20260113020000_v3.4.0.cdf",
    BASE_DIR / "mms1_fpi_fast_l2_dis-moms_20260113040000_v3.4.0.cdf",
]
REPORT_PATH = BASE_DIR / "analysis_summary.md"
SVG_PATH = BASE_DIR / "candidate_interval.svg"
CSV_PATH = BASE_DIR / "candidate_table.csv"

BIN_SECONDS = 5
VX_THRESHOLD = 300.0
MIN_CONSECUTIVE_BINS = 2
Bz_VAR_THRESHOLD = 3.0
STFT_WINDOW_SECONDS = 4.0
STFT_STEP_SECONDS = 1.0


@dataclass
class DatasetSummary:
    name: str
    start: pd.Timestamp
    end: pd.Timestamp
    rows: int
    cadence_seconds: float
    missing_count: int
    variable_note: str
    label_note: str


def load_epoch(cdf: cdflib.CDF) -> pd.DatetimeIndex:
    return pd.to_datetime(cdflib.cdfepoch.to_datetime(cdf.varget("Epoch")))


def cadence_seconds(index: pd.DatetimeIndex) -> float:
    return float(index.to_series().diff().dropna().dt.total_seconds().median())


def summarize_fgm() -> tuple[DatasetSummary, pd.DataFrame]:
    cdf = cdflib.CDF(FGM_FILE)
    epoch = load_epoch(cdf)
    b = np.asarray(cdf.varget("mms1_fgm_b_gse_srvy_l2"), dtype=np.float64)
    df = pd.DataFrame(
        {
            "time": epoch,
            "Bx": b[:, 0],
            "By": b[:, 1],
            "Bz": b[:, 2],
            "Bt": b[:, 3],
        }
    ).set_index("time")
    summary = DatasetSummary(
        name="FGM magnetic field",
        start=epoch.min(),
        end=epoch.max(),
        rows=len(epoch),
        cadence_seconds=cadence_seconds(epoch),
        missing_count=int(df.isna().sum().sum()),
        variable_note="timestamp + Bx/By/Bz/Bt in GSE",
        label_note="no BBF label; label_* fields are axis descriptors only",
    )
    return summary, df


def summarize_fpi() -> tuple[DatasetSummary, pd.DataFrame]:
    frames = []
    starts = []
    ends = []
    cadences = []
    for path in FPI_FILES:
        cdf = cdflib.CDF(path)
        epoch = load_epoch(cdf)
        v = np.asarray(cdf.varget("mms1_dis_bulkv_gse_fast"), dtype=np.float64)
        spectr = np.asarray(cdf.varget("mms1_dis_energyspectr_omni_fast"), dtype=np.float64)
        frames.append(
            pd.DataFrame(
                {
                    "time": epoch,
                    "Vx": v[:, 0],
                    "Vy": v[:, 1],
                    "Vz": v[:, 2],
                    "ion_spectr_omni_mean": np.nanmean(spectr, axis=1),
                }
            )
        )
        starts.append(epoch.min())
        ends.append(epoch.max())
        cadences.append(cadence_seconds(epoch))
    df = pd.concat(frames, ignore_index=True).sort_values("time").set_index("time")
    summary = DatasetSummary(
        name="FPI ion velocity",
        start=min(starts),
        end=max(ends),
        rows=len(df),
        cadence_seconds=float(np.median(cadences)),
        missing_count=int(df.isna().sum().sum()),
        variable_note="timestamp + Vx/Vy/Vz in GSE",
        label_note="no BBF label; *_label_fast fields are component names only",
    )
    return summary, df


def bandpower_from_signal(signal: np.ndarray, sample_rate_hz: float, f_low: float, f_high: float) -> float:
    """Estimate band-limited wave power from a short SCM segment.

    The Hanning window reduces spectral leakage before the FFT so the power
    estimate is less contaminated by sharp window edges. In this survey-rate
    feasibility script the goal is not a calibrated wave-normal analysis, but a
    stable low/mid/high-band proxy that can be compared across windows.
    """
    clean = np.nan_to_num(signal, nan=0.0)
    windowed = clean * np.hanning(len(clean))
    spectrum = np.fft.rfft(windowed)
    freqs = np.fft.rfftfreq(len(clean), d=1.0 / sample_rate_hz)
    power = np.abs(spectrum) ** 2
    mask = (freqs >= f_low) & (freqs < f_high)
    return float(power[mask].sum()) if mask.any() else float("nan")


def build_scm_features() -> tuple[DatasetSummary, pd.DataFrame]:
    """Derive survey-rate wave features that are actually observable in SCM data.

    The survey product only covers roughly 0.5-16 Hz here, so these features are
    feasibility proxies rather than true whistler-band measurements.
    """
    cdf = cdflib.CDF(SCM_FILE)
    epoch = load_epoch(cdf)
    w = np.asarray(cdf.varget("mms1_scm_acb_gse_scsrvy_srvy_l2"), dtype=np.float64)
    df = pd.DataFrame({"time": epoch, "Wx": w[:, 0], "Wy": w[:, 1], "Wz": w[:, 2]}).set_index("time")
    df["wave_rms"] = np.sqrt(df["Wx"] ** 2 + df["Wy"] ** 2 + df["Wz"] ** 2)

    dt = cadence_seconds(epoch)
    sample_rate_hz = 1.0 / dt
    window_n = max(8, int(round(STFT_WINDOW_SECONDS * sample_rate_hz)))
    step_n = max(1, int(round(STFT_STEP_SECONDS * sample_rate_hz)))
    times = []
    low_band = []
    mid_band = []
    high_band = []
    for start in range(0, len(df) - window_n + 1, step_n):
        segment = df["wave_rms"].to_numpy()[start : start + window_n]
        center_time = df.index[start + window_n // 2]
        times.append(center_time)
        low_band.append(bandpower_from_signal(segment, sample_rate_hz, 0.5, 2.0))
        mid_band.append(bandpower_from_signal(segment, sample_rate_hz, 2.0, 8.0))
        high_band.append(bandpower_from_signal(segment, sample_rate_hz, 8.0, 16.0))
    spec_df = pd.DataFrame(
        {
            "time": pd.DatetimeIndex(times),
            "wave_low_band_power": low_band,
            "wave_mid_band_power": mid_band,
            "wave_high_band_power": high_band,
        }
    ).set_index("time")

    summary = DatasetSummary(
        name="SCM wave",
        start=epoch.min(),
        end=epoch.max(),
        rows=len(df),
        cadence_seconds=dt,
        missing_count=int(df.isna().sum().sum()),
        variable_note="timestamp + raw waveform vectors; derived low/mid/high band powers in observable 0.5-16 Hz range",
        label_note="no BBF label",
    )
    return summary, spec_df


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
        return "tailward_candidate"
    return "none"


def build_joint_table(fpi_df: pd.DataFrame, fgm_df: pd.DataFrame, scm_df: pd.DataFrame) -> pd.DataFrame:
    """Combine FPI, FGM, and SCM features into a provisional BBF candidate table.

    This table mixes a simple operational BBF rule with wave-power proxies so the
    report can separate event flagging from later candidate ranking.
    """
    merged = (
        fpi_df.resample(f"{BIN_SECONDS}s").mean()
        .join(fgm_df[["Bx", "By", "Bz", "Bt"]].resample(f"{BIN_SECONDS}s").mean(), how="inner")
        .join(scm_df.resample(f"{BIN_SECONDS}s").mean(), how="inner")
    )
    merged["abs_vx"] = merged["Vx"].abs()
    merged["bbf_speed_flag"] = merged["abs_vx"] > VX_THRESHOLD
    merged["bbf_speed_run_flag"] = consecutive_true(merged["bbf_speed_flag"], MIN_CONSECUTIVE_BINS)
    merged["bz_delta"] = merged["Bz"].diff().abs()
    merged["bz_support_flag"] = merged["bz_delta"] > Bz_VAR_THRESHOLD
    merged["bbf_operational_flag"] = merged["bbf_speed_run_flag"] & merged["bz_support_flag"].fillna(False)
    merged["bbf_direction"] = merged["Vx"].apply(classify_vx_direction)

    merged["candidate_score"] = (
        robust_zscore(merged["abs_vx"]).abs()
        + robust_zscore(merged["bz_delta"].fillna(0.0)).abs()
        + robust_zscore(np.log10(merged["wave_high_band_power"].clip(lower=1.0))).abs()
    )

    merged["fce_hz"] = 28.0 * merged["Bt"]
    merged["observable_nyquist_hz"] = 0.5 / cadence_seconds(load_epoch(cdflib.CDF(SCM_FILE)))
    merged["whistler_low_hz"] = 0.1 * merged["fce_hz"]
    merged["whistler_high_hz"] = 0.5 * merged["fce_hz"]
    merged["whistler_observable_flag"] = merged["whistler_low_hz"] < merged["observable_nyquist_hz"]
    return merged


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
    panel_h = 150
    height = 660
    pad_x = 80
    pad_y = 24
    colors = {"Vx": "#0b6e4f", "Bz": "#b22222", "wave_high_band_power": "#1f4e79", "candidate_score": "#6b4f9d"}
    panels = [
        ("Vx", "Velocity Vx [km/s]"),
        ("Bz", "Magnetic Field Bz [nT]"),
        ("wave_high_band_power", "SCM High-Band Power [8-16 Hz]"),
        ("candidate_score", "Candidate Score"),
    ]
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{pad_x}" y="28" font-size="22" font-family="Arial">BBF candidate interval</text>',
        f'<text x="{pad_x}" y="50" font-size="14" font-family="Arial">{start} to {end} ({BIN_SECONDS} s bins)</text>',
    ]
    for idx, (col, title) in enumerate(panels):
        top = 70 + idx * (panel_h + 10)
        bottom = top + panel_h
        values = window[col].to_numpy(dtype=float)
        points = scale_points(values, width, panel_h, pad_x, pad_y, top)
        parts.append(f'<rect x="{pad_x}" y="{top}" width="{width - 2 * pad_x}" height="{panel_h}" fill="none" stroke="#cccccc"/>')
        parts.append(f'<text x="{pad_x}" y="{top - 8}" font-size="16" font-family="Arial">{title}</text>')
        parts.append(f'<polyline fill="none" stroke="{colors[col]}" stroke-width="2" points="{points}"/>')
        parts.append(f'<text x="10" y="{top + 16}" font-size="12" font-family="Arial">{np.nanmax(values):.2f}</text>')
        parts.append(f'<text x="10" y="{bottom - 6}" font-size="12" font-family="Arial">{np.nanmin(values):.2f}</text>')
    parts.append("</svg>")
    with SVG_PATH.open("w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))


def write_report(summaries: list[DatasetSummary], merged: pd.DataFrame) -> None:
    operational_hits = merged[merged["bbf_operational_flag"]].copy()
    if operational_hits.empty:
        candidate_time = merged["candidate_score"].idxmax()
    else:
        candidate_time = operational_hits["candidate_score"].idxmax()

    top_candidates = merged.sort_values("candidate_score", ascending=False).head(10).copy()
    top_candidates = top_candidates.reset_index()
    top_candidates.to_csv(CSV_PATH, index=False)

    window_start = candidate_time - pd.Timedelta(minutes=5)
    window_end = candidate_time + pd.Timedelta(minutes=10)
    render_svg(merged.loc[window_start:window_end, ["Vx", "Bz", "wave_high_band_power", "candidate_score"]], window_start, window_end)

    lines = [
        "# MMS Analysis Summary",
        "",
        "## Operational Definition",
        f"- Bin size: `{BIN_SECONDS}` seconds.",
        f"- Speed rule: `|Vx| > {VX_THRESHOLD:.0f} km/s`.",
        f"- Persistence rule: at least `{MIN_CONSECUTIVE_BINS}` consecutive bins.",
        f"- Magnetic support rule: `|delta Bz| > {Bz_VAR_THRESHOLD:.0f} nT` in the same bin.",
        "- Candidate ranking is separate from event flagging. `candidate_score` ranks unusual windows; `bbf_operational_flag` is the provisional event rule.",
        "",
        "## Direction Convention",
        "- `Vx > 300`: provisional earthward BBF candidate.",
        "- `Vx < -300`: provisional tailward BBF candidate.",
        "- Current strongest candidates in this dataset are tailward because the largest-magnitude Vx values are negative.",
        "",
        "## Dataset Check",
        "| Dataset | Time Range | Rows | Median Cadence | Missing Values | Variables | BBF Label |",
        "|---|---|---:|---:|---:|---|---|",
    ]

    for item in summaries:
        lines.append(
            f"| {item.name} | {item.start} to {item.end} | {item.rows} | {item.cadence_seconds:.6f}s | "
            f"{item.missing_count} | {item.variable_note} | {item.label_note} |"
        )

    lines.extend(
        [
            "",
            "## Raw vs Feature Table",
            "- These files are raw or near-raw time-series products, not a prebuilt BBF feature table.",
            "- `timestamp` exists across all products.",
            "- `Vx/Vy/Vz` exists in FPI.",
            "- `Bx/By/Bz` exists in FGM.",
            "- Wave content exists in SCM, but only as survey-rate waveform. There is no BBF label column.",
            "",
            "## Whistler Feasibility",
            f"- SCM survey sample rate is about `{1.0 / summaries[2].cadence_seconds:.2f} Hz`, so Nyquist is about `{0.5 / summaries[2].cadence_seconds:.2f} Hz`.",
            "- FGM total field implies `fce ≈ 28 * Bt[nT]`, which is hundreds of Hz here.",
            "- Therefore the classic whistler band is not observable in this SCM survey product. The script derives observable low/mid/high band powers in `0.5-16 Hz`, but that is not a true whistler-band measurement.",
            "",
            "## Candidate Generation",
            f"- Operational BBF bins found: `{int(merged['bbf_operational_flag'].sum())}`",
            f"- Earthward candidate bins: `{int((merged['bbf_direction'] == 'earthward_candidate').sum())}`",
            f"- Tailward candidate bins: `{int((merged['bbf_direction'] == 'tailward_candidate').sum())}`",
            "",
            "| Time | Vx | Bz | High-Band Power | candidate_score | operational_flag | direction |",
            "|---|---:|---:|---:|---:|---|---|",
        ]
    )

    for _, row in top_candidates.iterrows():
        lines.append(
            f"| {row['time']} | {row['Vx']:.3f} | {row['Bz']:.3f} | {row['wave_high_band_power']:.3f} | "
            f"{row['candidate_score']:.3f} | {bool(row['bbf_operational_flag'])} | {row['bbf_direction']} |"
        )

    lines.extend(
        [
            "",
            "## Label Separation",
            "- `candidate_score`: unsupervised ranking for candidate generation.",
            "- `bbf_operational_flag`: rule-based provisional event flag.",
            "- `actual_bbf_label`: unavailable in these files and must come from later labeling or an external event list.",
            "",
            f"Primary candidate center time: `{candidate_time}`",
            f"Candidate table: `{CSV_PATH}`",
            f"Candidate plot: `{SVG_PATH}`",
        ]
    )

    with REPORT_PATH.open("w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))


def main() -> None:
    fpi_summary, fpi_df = summarize_fpi()
    fgm_summary, fgm_df = summarize_fgm()
    scm_summary, scm_df = build_scm_features()
    merged = build_joint_table(fpi_df, fgm_df, scm_df)
    write_report([fpi_summary, fgm_summary, scm_summary], merged)
    print(f"report: {REPORT_PATH}")
    print(f"plot: {SVG_PATH}")
    print(f"candidate csv: {CSV_PATH}")


if __name__ == "__main__":
    main()
