# Event Coupling Summary

## Analysis Rules
- BBF speed rule: `Vx > +300 km/s`, at least `2` consecutive `5s` bins.
- Tailward fast flow is tracked separately as `Vx < -300 km/s` and is excluded from the BBF set.
- BBF magnetic support is evaluated at event level: `max |delta Bz| > 3 nT` somewhere inside the speed-defined event.
- Whistler seed segment rule: `whistler_score >= q75` (`1.928`), `fce >= 50 Hz`, and `background_excess >= max(q75, 3.0)` (`1.502`).
- Whistler event filters: minimum duration `0.25s`, band occupancy `>= 0.60`, peak-frequency CV `<= 0.35`, merge gaps up to `1.0s`.
- Leading: `-30s` to `-5s` before BBF start.
- Coincident: within `+/-5s` of BBF start.
- Lagging: `+5s` to `+60s` after BBF start.
- Other offsets are `uncorrelated`.

## Event Counts
- Total earthward BBF events: `4`
- Total whistler events: `32`
- Tailward fast-flow runs excluded from BBF set: `0`
- Tailward events surviving BBF filter: `0`
- Earthward BBF events: `4`

## Lag Statistics
- Mean onset lag: `32.241s`
- Median onset lag: `38.497s`
- Mean lag relative to BBF front time: `-65.259s`
- Median lag relative to BBF front time: `-66.503s`
- Mean lag relative to BBF peak Vx: `-12.759s`
- Mean lag relative to max |dBz| time: `-116.509s`
- Leading: `0`
- Coincident: `1`
- Lagging: `3`
- Uncorrelated: `0`
- Front-referenced coincident: `0`

## Event Phase
- Median normalized phase `phi`: `0.120`
- Pre-event (`phi < 0`): `0`
- Early (`0 <= phi < 1/3`): `3`
- Mid (`1/3 <= phi < 2/3`): `1`
- Late (`2/3 <= phi <= 1`): `0`
- Post-event (`phi > 1`): `0`

## Overlap And Baseline
- Analysis interval: `2017-07-29 15:13:03.248102158` to `2017-07-29 16:08:40.290771492` (669 bins)
- BBF occupancy: `222/669` (`0.3318`)
- Whistler occupancy: `34/669` (`0.0508`)
- Actual overlap bins: `33`
- Expected random overlap: `11.283`
- Monte Carlo expected overlap: `11.329`
- Permutation p-value: `0.0005`
- Overlap enrichment: `2.925x` / `2.913x` (Monte Carlo)

## Conditional Probability
- Within 10 s: observed `0.250` vs baseline `0.091` -> `2.734x`
- Within 30 s: observed `0.500` vs baseline `0.250` -> `2.000x`
- Within 60 s: observed `1.000` vs baseline `0.437` -> `2.286x`

## BBF To Nearest Whistler
| BBF Event | Start | End | Direction | Peak Vx | Max |dBz| | Nearest Whistler Onset | Start Lag [s] | Front Lag [s] | Vx Peak Lag [s] | |dBz| Peak Lag [s] | Phi | Phase | Class | Front Class | Overlap |
|---|---|---|---|---:|---:|---|---:|---:|---:|---:|---:|---|---|---|---|
| 1 | 2017-07-29 15:47:35 | 2017-07-29 15:48:55 | earthward | 858.7 | 5.30 | 2017-07-29 15:48:23.431918054 | 48.43 | 18.43 | -11.57 | 13.43 | 0.605 | mid | lagging | lagging | True |
| 2 | 2017-07-29 15:49:10 | 2017-07-29 15:53:20 | earthward | 868.0 | 3.80 | 2017-07-29 15:49:11.682186843 | 1.68 | -173.32 | -23.32 | -173.32 | 0.007 | early | coincident | uncorrelated | True |
| 3 | 2017-07-29 15:55:25 | 2017-07-29 16:03:50 | earthward | 997.7 | 5.82 | 2017-07-29 15:55:53.561574316 | 28.56 | -151.44 | -46.44 | -351.44 | 0.057 | early | lagging | uncorrelated | True |
| 4 | 2017-07-29 16:03:55 | 2017-07-29 16:08:30 | earthward | 975.6 | 3.49 | 2017-07-29 16:04:45.287603457 | 50.29 | 45.29 | 30.29 | 45.29 | 0.183 | early | lagging | lagging | True |

## Threshold Sweep
- Background-excess sweep CSV: `C:\Magnetic\cases\2017-07-29_mms1_earthward_bbf\threshold_sweep.csv`

BBF events CSV: `C:\Magnetic\cases\2017-07-29_mms1_earthward_bbf\bbf_events.csv`
Whistler events CSV: `C:\Magnetic\cases\2017-07-29_mms1_earthward_bbf\whistler_events.csv`
Coupling CSV: `C:\Magnetic\cases\2017-07-29_mms1_earthward_bbf\event_coupling.csv`