# Baseline Santolik-Style Whistler Detector

This workspace is isolated from the current `background_excess` detector so the two pipelines can be compared without overwriting prior outputs.

## Baseline Rule

The baseline detector will retain only wave power that satisfies all of the following:

- `0.1 fce <= f_peak <= 0.5 fce`
- `ellipticity > 0.7`
- `planarity > 0.7`
- `magnetic PSD > 1e-7 nT^2/Hz`

This is the lightweight Santolik-style baseline discussed for the BBF comparison study.

## Intended Outputs

- `baseline_segments.csv`
- `baseline_events.csv`
- `baseline_summary.md`
- `baseline_vs_current_notes.md`

## Current Status

The folder and config are prepared.
The next implementation step is the spectral-matrix SVD polarization calculation from SCM burst waveforms.
