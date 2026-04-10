# 2017-07-29 MMS1 Earthward BBF Case

이 폴더는 `2017-07-29` MMS1 earthward BBF 사례를 별도로 분석하기 위한 작업 공간이다.

## Target Interval
- Start: `2017-07-29 15:00:00 UTC`
- Stop: `2017-07-29 17:00:00 UTC`

## Goal
- `earthward BBF (Vx > +300 km/s)` 구간 확인
- BBF event 추출
- SCM burst 기반 whistler event 추출
- BBF-whistler lag / coupling / baseline 계산

## Suggested MMS1 Products
- FPI ion moments: `mms1_fpi_*_dis-moms`
- FGM magnetic field: `mms1_fgm_*`
- SCM burst waveform: `mms1_scm_brst_*`

## Notes
- 이 케이스에서는 `tailward fast flow`를 BBF와 분리한다.
- BBF 정의는 `earthward only`로 유지한다.
- 기존 `C:\Magnetic` 루트의 2026 파일들과 섞지 않는다.
