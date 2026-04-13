# MMS BBF-Whistler Coupling Analysis

MMS1 관측 자료에서 earthward bursty bulk flow(BBF)와 whistler-mode wave 후보의 시간적 결합을 분석한 작업 저장소입니다. 핵심 케이스는 `2017-07-29 15:00:00-17:00:00 UTC` 구간이며, BBF 검출, SCM burst 기반 whistler 후보 추출, Santolik-style baseline 검증, BBF-whistler lag/overlap 통계를 함께 산출합니다.

## 핵심 결과

- 분석 케이스: MMS1 `2017-07-29` earthward BBF interval.
- BBF 정의: `Vx > +300 km/s`, `5 s` bin에서 최소 `2`개 연속 bin.
- event-level 자기장 support: speed-defined BBF event 내부에서 `max |delta Bz| > 3 nT`.
- 검출된 earthward BBF event: `4`.
- 검출된 whistler event: `32`.
- model-ready whistler feature row: `4264`.
- Santolik-style strict segment label: `247`.
- Santolik-style strict event label: `454`.
- BBF-whistler overlap enrichment: `2.925x` observed / `2.913x` Monte Carlo.
- permutation p-value: `0.0005`.
- BBF 시작 기준 whistler onset class: coincident `1`, lagging `3`, leading `0`, uncorrelated `0`.

## 이벤트 결합 요약

| BBF Event | Start | End | Peak Vx [km/s] | Max \|dBz\| [nT] | Nearest Whistler Onset | Start Lag [s] | Phase | Class | Overlap |
|---:|---|---|---:|---:|---|---:|---|---|---|
| 1 | 2017-07-29 15:47:35 | 2017-07-29 15:48:55 | 858.7 | 5.30 | 2017-07-29 15:48:23.431918054 | 48.43 | mid | lagging | true |
| 2 | 2017-07-29 15:49:10 | 2017-07-29 15:53:20 | 868.0 | 3.80 | 2017-07-29 15:49:11.682186843 | 1.68 | early | coincident | true |
| 3 | 2017-07-29 15:55:25 | 2017-07-29 16:03:50 | 997.7 | 5.82 | 2017-07-29 15:55:53.561574316 | 28.56 | early | lagging | true |
| 4 | 2017-07-29 16:03:55 | 2017-07-29 16:08:30 | 975.6 | 3.49 | 2017-07-29 16:04:45.287603457 | 50.29 | early | lagging | true |

Conditional probability는 모든 window에서 random baseline보다 높았습니다.

| Window | Observed | Baseline | Enrichment |
|---:|---:|---:|---:|
| 10 s | 0.250 | 0.091 | 2.734x |
| 30 s | 0.500 | 0.250 | 2.000x |
| 60 s | 1.000 | 0.437 | 2.286x |

## Whistler 분석

SCM burst waveform의 sample rate는 약 `16383.77 Hz`로, 이 케이스에서는 whistler band를 직접 분석할 수 있습니다. Operational whistler band는 FGM의 `Bt`에서 계산한 local electron cyclotron frequency를 기준으로 동적으로 정의했습니다.

- Dynamic whistler band: `0.1-0.5 fce`.
- `fce ~= 28 * Bt[nT]`.
- Whistler seed rule: `whistler_score >= q75`, `fce >= 50 Hz`, `background_excess >= max(q75, 3.0)`.
- Event filter: duration `>= 0.25 s`, band occupancy `>= 0.60`, peak-frequency CV `<= 0.35`, merge gap `<= 1.0 s`.
- 가장 강한 whistler segment: `2017-07-29 15:59:04.471211044`.

Santolik-style baseline은 다음 기준으로 별도 strict label을 만들었습니다.

- Frequency gate: `0.1 fce` to `0.5 fce`.
- Ellipticity `> 0.7`.
- Planarity `> 0.7`.
- PSD `> 1e-07 nT^2/Hz`.
- Evaluated segments: `3758`.
- Passing baseline segments: `224`.
- Baseline events: `40`.
- Current detector events: `26`.

## 저장소 구조

```text
.
├── analyze_mms.py
├── analyze_whistler_burst.py
├── analyze_whistler_baseline.py
├── analyze_event_coupling.py
├── analysis_summary.md
└── cases/
    └── 2017-07-29_mms1_earthward_bbf/
        ├── README.md
        ├── case_config.json
        ├── whistler_burst_summary.md
        ├── event_coupling_summary.md
        ├── bbf_events.csv
        ├── whistler_events.csv
        ├── event_coupling.csv
        ├── whistler_model_features.csv
        ├── threshold_sweep.csv
        ├── whistler_burst_overview.png
        └── baseline_santolik/
            ├── baseline_summary.md
            ├── baseline_segments.csv
            ├── baseline_events.csv
            └── baseline_overview.png
```

Raw MMS CDF 자료는 용량과 재현성 관리를 위해 Git 추적 대상에서 제외되어 있습니다. 케이스별 원자료는 보통 `cases/<case>/data/` 아래에 둡니다.

## 실행 방법

Python 의존성은 `cdflib`, `numpy`, `pandas`, `matplotlib`가 필요합니다.

```powershell
.\venv\Scripts\Activate.ps1
$env:MMS_CASE_DIR = "C:\Magnetic\cases\2017-07-29_mms1_earthward_bbf"
python analyze_whistler_burst.py
python analyze_whistler_baseline.py
python analyze_event_coupling.py
```

스크립트는 `MMS_CASE_DIR` 환경 변수를 사용합니다. 지정하지 않으면 기본값은 `C:\Magnetic`입니다.

## 주요 산출물

- BBF event table: `cases/2017-07-29_mms1_earthward_bbf/bbf_events.csv`
- Whistler event table: `cases/2017-07-29_mms1_earthward_bbf/whistler_events.csv`
- BBF-whistler coupling table: `cases/2017-07-29_mms1_earthward_bbf/event_coupling.csv`
- Model-ready feature table: `cases/2017-07-29_mms1_earthward_bbf/whistler_model_features.csv`
- Threshold sweep: `cases/2017-07-29_mms1_earthward_bbf/threshold_sweep.csv`
- Whistler overview plot: `cases/2017-07-29_mms1_earthward_bbf/whistler_burst_overview.png`
- Santolik baseline overview plot: `cases/2017-07-29_mms1_earthward_bbf/baseline_santolik/baseline_overview.png`

## 초기 survey-rate 점검

루트의 `analysis_summary.md`는 `2026-01-13` survey-rate 자료를 대상으로 한 초기 feasibility check입니다. 이 자료에서는 SCM survey sample rate가 약 `32 Hz`라 Nyquist가 약 `16 Hz`에 머물러 classic whistler band를 직접 볼 수 없었습니다. 따라서 현재 연구의 주 분석은 SCM burst 자료가 있는 `2017-07-29` 케이스로 분리했습니다.

## 해석상 주의점

- `whistler_score`는 candidate-generation score이며 최종 label이 아닙니다.
- `bbf_operational_flag`는 rule-based provisional event flag입니다.
- Tailward fast flow는 별도 추적하되, 이 케이스의 earthward BBF set에서는 제외했습니다.
- Strict Santolik-style label은 모델 학습/검증용 boolean target으로 붙였고, continuous predictor는 thresholding하지 않은 값으로 보존했습니다.
