# MMS BBF-Whistler Coupling Analysis

MMS1 관측 자료에서 earthward bursty bulk flow(BBF)와 whistler-mode wave 후보의 시간적 결합을 분석하는 작업 저장소입니다. 현재 로컬 분석은 `2017-07-29` MMS1 burst 자료를 중심으로 두 단계로 정리되어 있습니다.

- 초기 집중 케이스: `cases/2017-07-29_mms1_earthward_bbf`, `2017-07-29 15:00:00-17:00:00 UTC`.
- 확장 하루 케이스: `cases/2017-07-29_mms1_full_day`, 설정상 `2017-07-29T00:00:00Z`부터 `2017-07-29T23:59:59Z`.

주의: MMS burst 자료는 하루 전체를 연속으로 덮지 않습니다. Full-day 케이스는 해당 날짜 manifest에 있는 burst interval 41개를 모두 합친 케이스이며, 실제 공통 분석 구간은 `2017-07-29 02:25:23.231949915`부터 `2017-07-29 20:35:10.721812125`까지입니다.

## 현재 핵심 결과

Full-day 케이스 기준:

- 다운로드 완료 raw CDF: `124/124` 파일, 약 `1.023 GiB`.
- Products: `fgm_srvy`, `fgm_brst`, `fpi_brst_dis_moms`, `scm_brst_schb`.
- SCM burst files analyzed: `41`.
- Whistler feature/model rows: `22087`.
- Detected whistler events: `135`.
- Santolik-style evaluated segments: `19491`.
- Santolik-style passing segments: `1254`.
- Santolik-style baseline events: `230`.
- Earthward BBF events: `8`.
- BBF-whistler lag class: leading `1`, coincident `2`, lagging `5`, uncorrelated `0`.
- BBF-whistler overlap enrichment: `9.993x` observed / `9.562x` Monte Carlo.
- Permutation p-value: `0.0005`.
- Within 60 s conditional probability: observed `0.875` vs baseline `0.141`, enrichment `6.193x`.

## Full-Day BBF Events

| Event | Start | End | Duration [s] | Peak Vx [km/s] | Max \|dBz\| [nT] |
|---:|---|---|---:|---:|---:|
| 1 | 2017-07-29 08:17:30 | 2017-07-29 08:19:25 | 115 | 669.5 | 3.28 |
| 2 | 2017-07-29 08:20:50 | 2017-07-29 08:24:00 | 190 | 756.2 | 7.00 |
| 3 | 2017-07-29 09:24:55 | 2017-07-29 09:27:05 | 130 | 802.0 | 3.96 |
| 4 | 2017-07-29 10:24:55 | 2017-07-29 10:25:10 | 15 | 419.3 | 5.15 |
| 5 | 2017-07-29 15:47:35 | 2017-07-29 15:48:55 | 80 | 858.7 | 5.30 |
| 6 | 2017-07-29 15:49:10 | 2017-07-29 15:53:20 | 250 | 868.0 | 3.80 |
| 7 | 2017-07-29 15:55:25 | 2017-07-29 16:03:50 | 505 | 997.7 | 5.82 |
| 8 | 2017-07-29 16:03:55 | 2017-07-29 16:08:30 | 275 | 975.6 | 3.49 |

## Early BBF Model Pilot

Full-day 케이스에서 early BBF prediction용 causal dataset을 만들고 Leave-One-Out CV baseline 모델을 돌렸습니다. 목표는 anchor time `t` 이후 `300 s` 안에 BBF start가 있는지 예측하는 것입니다. 입력 window는 `t-900 s`부터 `t`까지이며, `Vx`와 BBF label 계열은 입력에서 제외했습니다.

전체 anchor 평가:

| Model | Samples | Positive | Accuracy | Balanced Acc | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|---:|
| majority | 222 | 15 | 0.932 | 0.500 | 0.000 | 0.500 |
| logistic_regression | 222 | 15 | 0.874 | 0.809 | 0.440 | 0.840 |
| random_forest | 222 | 15 | 0.914 | 0.738 | 0.457 | 0.932 |
| xgboost | 222 | 15 | 0.919 | 0.493 | 0.000 | 0.941 |

BBF 내부 anchor 제거 평가:

| Model | Samples | Positive | Accuracy | Balanced Acc | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|---:|
| majority | 216 | 12 | 0.944 | 0.500 | 0.000 | 0.500 |
| logistic_regression | 216 | 12 | 0.833 | 0.833 | 0.357 | 0.845 |
| random_forest | 216 | 12 | 0.935 | 0.770 | 0.500 | 0.878 |
| xgboost | 216 | 12 | 0.940 | 0.498 | 0.000 | 0.886 |

해석상 아직 pilot 수준입니다. Positive sample이 `12-15`개뿐이므로, 이 결과는 모델 선택의 근거라기보다 데이터 파이프라인과 feature leakage 점검용 baseline으로 봐야 합니다.

## 저장소 구조

```text
.
├── analyze_whistler_burst.py
├── analyze_whistler_baseline.py
├── analyze_event_coupling.py
├── build_ml_dataset.py
├── build_early_bbf_dataset.py
├── evaluate_early_bbf_pilot.py
├── build_yearly_ml_dataset.py
├── build_yearly_bbf_candidates.py
├── download_mms_day.py
├── mms_sdc_manifest.py
├── manifests/
│   ├── mms1_2017-01-01_to_2018-01-01_manifest.csv
│   └── mms1_2017-01-01_to_2018-01-01_summary.json
├── datasets/
│   └── yearly_2017_mms1/
└── cases/
    ├── 2017-07-29_mms1_earthward_bbf/
    └── 2017-07-29_mms1_full_day/
```

Raw MMS CDF 자료는 용량과 재현성 관리를 위해 Git 추적 대상에서 제외합니다. 케이스별 원자료는 `cases/<case>/data/` 아래에 둡니다.

## 주요 산출물

Full-day 케이스:

- Case README: `cases/2017-07-29_mms1_full_day/README.md`
- Download summary: `cases/2017-07-29_mms1_full_day/download_summary.json`
- BBF events: `cases/2017-07-29_mms1_full_day/bbf_events.csv`
- Whistler events: `cases/2017-07-29_mms1_full_day/whistler_events.csv`
- Coupling summary: `cases/2017-07-29_mms1_full_day/event_coupling_summary.md`
- Coupling table: `cases/2017-07-29_mms1_full_day/event_coupling.csv`
- Model-ready feature table: `cases/2017-07-29_mms1_full_day/whistler_model_features.csv`
- ML tabular dataset: `cases/2017-07-29_mms1_full_day/ml_dataset/tabular_features.parquet`
- ML sequence index: `cases/2017-07-29_mms1_full_day/ml_dataset/sequence_index.csv`
- Early BBF dataset: `cases/2017-07-29_mms1_full_day/early_bbf_dataset/`
- Early BBF model results: `cases/2017-07-29_mms1_full_day/early_bbf_dataset/reports/baseline_results_clean.csv`

Original 2-hour 케이스:

- Case README: `cases/2017-07-29_mms1_earthward_bbf/README.md`
- Coupling summary: `cases/2017-07-29_mms1_earthward_bbf/event_coupling_summary.md`
- ML dataset: `cases/2017-07-29_mms1_earthward_bbf/ml_dataset/`

Year-range pilot:

- Year manifest: `manifests/mms1_2017-01-01_to_2018-01-01_manifest.csv`
- Year manifest summary: `manifests/mms1_2017-01-01_to_2018-01-01_summary.json`
- FPI-fast BBF speed candidates: `datasets/yearly_2017_mms1/bbf_speed_candidates_2017.csv`

## Reproduce Full-Day Case

```powershell
.\venv\Scripts\Activate.ps1

python download_mms_day.py `
  --date 2017-07-29 `
  --case-id 2017-07-29_mms1_full_day `
  --max-download-gib 2.0

$env:MMS_CASE_DIR = "C:\Magnetic\cases\2017-07-29_mms1_full_day"
python analyze_whistler_burst.py
python analyze_whistler_baseline.py
python analyze_event_coupling.py
python build_ml_dataset.py
python build_early_bbf_dataset.py --case-dir C:\Magnetic\cases\2017-07-29_mms1_full_day --max-sequences 0
python evaluate_early_bbf_pilot.py --dataset-dir C:\Magnetic\cases\2017-07-29_mms1_full_day\early_bbf_dataset
python evaluate_early_bbf_pilot.py --dataset-dir C:\Magnetic\cases\2017-07-29_mms1_full_day\early_bbf_dataset --drop-inside-bbf
```

## Year-Range Notes

`mms_sdc_manifest.py`로 만든 manifest는 `2017-01-01`부터 `2018-01-01`까지를 대상으로 합니다. Manifest 기준 2017년 full raw 크기는 대략 다음과 같습니다.

- FGM survey: `23.536 GiB`
- FPI fast ion moments: `3.912 GiB`
- FPI burst ion moments: `9.906 GiB`
- FGM burst: `11.596 GiB`
- SCM burst: `104.722 GiB`

따라서 전체 SCM burst raw를 한 번에 저장하기보다, FPI fast로 후보를 먼저 찾고 후보 주변 burst 자료를 별도 케이스로 내려받는 방식이 현실적입니다. 현재 `build_yearly_bbf_candidates.py --resume` pilot은 `2017-01-01`부터 `2017-01-07`까지 처리했고 speed candidate `181`개를 추출했습니다.

## 해석상 주의점

- `whistler_score`는 candidate-generation score이며 최종 label이 아닙니다.
- `bbf_operational_flag`는 rule-based provisional event flag입니다.
- Santolik-style label은 모델 학습/검증용 strict boolean target으로 붙인 별도 기준입니다.
- Full-day 케이스는 하루 전체 continuous coverage가 아니라 2017-07-29에 존재하는 MMS burst interval 묶음입니다.
- 현재 모델 평가는 단일 날짜 pilot이므로 일반화 성능으로 해석하면 안 됩니다.
