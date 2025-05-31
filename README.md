# 아파트 주차장 EV 충전 시뮬레이션

Python과 SimPy를 사용한 아파트 주차장 내 전기차 충전 시뮬레이션 엔진입니다.


## 실행 방법
python  main.py --animation
-> 2시간 짜리 기본 시뮬레이션에 대한 로그와 영상 저장됨.

## 개요

이 시뮬레이션은 아파트 주차장에서 일반 차량과 전기차의 주차, 충전, 출차 과정을 모델링합니다. 차량은 입구에서 반시계 방향 일방통행 도로를 따라 순회하며 빈 일반 주차면 또는 EV 충전소에 주차하고, 주차 시간이 지나면 출차합니다.

## 주요 기능

- 주차장 맵 정의 (10×6 그리드)
- 일반 차량 및 전기차 모델링
- 확률적 도착 시간, 주차 시간, 배터리 잔량 및 충전 시간
- 이벤트 로깅 및 분석
- 주차장 상태 시각화
- 주차장 상태 애니메이션 생성 및 저장

## 설치 방법

### 의존성 설치

```bash
pip install -r requirements.txt
```

## 실행 방법

기본 설정으로 시뮬레이션 실행:

```bash
python main.py
```

사용자 정의 매개변수로 실행:

```bash
python main.py --seed 42 --time 86400 --normal 25 --ev 5 --visualize --save-csv
```

### 주요 옵션

- `--seed`: 난수 생성기 시드
- `--time`: 시뮬레이션 시간 (초)
- `--normal`: 일반 차량 수
- `--ev`: 전기차 수
- `--parking-capacity`: 일반 주차면 용량
- `--charger-capacity`: EV 충전소 용량
- `--visualize`: 시뮬레이션 결과 시각화 활성화
- `--save-csv`: 결과를 CSV 파일로 저장
- `--output-prefix`: 출력 파일 이름 접두사

### 애니메이션 생성

시뮬레이션 실행 후 생성된 로그 파일로 주차장 상태의 애니메이션을 생성할 수 있습니다:

```bash
python parking_animation.py results_sim_YYYYMMDD_HHMMSS/simulation_log.csv
```

사용자 정의 매개변수로 애니메이션 생성:

```bash
python parking_animation.py results_sim_YYYYMMDD_HHMMSS/simulation_log.csv custom_animation.mp4 --fps 15 --dpi 150 --speed 120
```

#### 애니메이션 옵션

- `log_file`: 시뮬레이션 로그 CSV 파일 경로 (필수)
- `output_file`: 저장할 애니메이션 파일 경로 (기본값: parking_animation.mp4)
- `--fps`: 애니메이션 프레임 속도 (기본값: 10)
- `--dpi`: 이미지 해상도 (기본값: 100)
- `--speed`: 시뮬레이션 속도 (실제 1초당 시뮬레이션 시간 초, 기본값: 60.0)

## 프로젝트 구조

```
.
├── main.py                 # 메인 실행 파일
├── parking_animation.py    # 주차장 애니메이션 생성 도구
├── requirements.txt        # 의존성 목록
└── src/                    # 소스코드 디렉토리
    ├── config.py           # 설정 상수
    ├── models/             # 시뮬레이션 모델
    │   ├── generator.py    # 차량 생성기
    │   ├── simulation.py   # 시뮬레이션 관리자
    │   └── vehicle.py      # 차량 모델
    └── utils/              # 유틸리티
        ├── helpers.py      # 헬퍼 함수
        ├── logger.py       # 로깅 시스템
        └── visualizer.py   # 시각화 도구
```

## 주차장 맵 정의

맵은 다음 문자로 정의됩니다:
- `N`: 경계/미사용 공간
- `E`: 입구/출구
- `R`: 도로
- `P`: 일반 주차면
- `C`: EV 충전소

## 확장 및 사용자 정의

시뮬레이션 파라미터, 샘플링 분포, 주차장 맵 등은 `src/config.py`에서 수정할 수 있습니다.
사용자 정의 분포는 `CustomParkingSimulation` 클래스를 통해 구현할 수 있습니다.

## 출력 결과

- 콘솔에 요약 정보 출력
- 시간대별 차량 도착 그래프 (`arrivals_by_hour.png`)
- 차량 유형별 주차 시간 분포 그래프 (`parking_duration.png`)
- 주차장 상태 스냅샷 이미지 (시작, 중간, 종료 시점)
- 주차장 상태 변화 애니메이션 비디오 (parking_animation.mp4)

## 최근 업데이트 내용

### 1. 차량 도착 순서 랜덤화
- 일반 차량과 전기차가 랜덤한 순서로 도착하도록 개선
- 실제 주차장 환경을 더 현실적으로 모델링

### 2. 전기차 충전 로직 개선
- 전기차의 충전 과정을 실시간으로 시뮬레이션
- 5분에 1%씩 배터리 충전 속도 적용
- 충전 과정에서 10% 단위로 배터리 상태 로깅

### 3. 충전소 우선 탐색
- 전기차는 충전소를 우선적으로 탐색하도록 로직 개선
- 배터리 부족 상태(50% 이하)로 도착하여 충전 필요성 증가

### 4. 충전 로그 시스템
- 충전 관련 이벤트를 별도로 기록하는 로그 시스템 구현
- 충전 시작, 업데이트, 종료 이벤트 추적
- 충전 패턴 분석을 위한 그래프 생성

### 5. 데이터 분석 기능
- 시뮬레이션 결과를 분석하는 Jupyter Notebook 추가
- 차량 이동 패턴, 주차 시간, 충전소 이용 현황 등 분석
- 데이터 시각화를 통한 인사이트 도출

### 6. 주차장 상태 애니메이션 기능
- 시뮬레이션 로그를 기반으로 주차장 상태 변화를 애니메이션으로 시각화
- 시간에 따른 차량 이동, 주차, 충전 상태를 동적으로 표현
- 애니메이션 속도, 해상도 등 사용자 정의 가능
- MP4 비디오 형식으로 결과 저장 및 공유 가능

## 사용 방법

```bash
# 기본 실행
python main.py

# 차량 수와 충전소 수 지정
python main.py --normal 10 --ev 5 --charger-capacity 3

# 시각화 및 CSV 저장
python main.py --visualize --save-csv

# 실행 시간 설정 (초 단위)
python main.py --time 7200  # 2시간 시뮬레이션

# 애니메이션 생성 (시뮬레이션 후)
python parking_animation.py results_sim_YYYYMMDD_HHMMSS/simulation_log.csv
```

## 분석 방법

1. 시뮬레이션 실행 후 생성된 CSV 파일 확인
2. Jupyter Notebook 실행: `jupyter notebook simulation_analysis.ipynb`
3. 생성된 그래프 및 시각화 자료 분석
4. 애니메이션 결과물로 시간에 따른 주차장 상태 변화 확인 # SME_Capstone_Team4

# commit test

## 현재 이슈
- 애니메이션에서 차량이 도로를 이동할 때 해당 셀을 연보라색으로 표시하는 기능이 아직 완벽하게 구현되지 않았습니다. (2024-05-22)

## 최근 업데이트 (2024-05-22)
1. 시뮬레이션 결과 저장 경로 변경
   - 결과 파일들이 상위 디렉토리에 저장되도록 수정
   - 디렉토리 이름 형식: `results_sim_YYYYMMDD_HHMMSS`

2. CSV 자동 저장 기능 추가
   - 시뮬레이션 결과가 기본적으로 CSV로 저장됨
   - `--no-save-csv` 옵션으로 CSV 저장 비활성화 가능

3. 한글 폰트 호환성 개선
   - 운영체제별 한글 폰트 자동 설정
   - Windows: 'Malgun Gothic' (맑은 고딕)
   - macOS: 'AppleGothic' (애플 고딕)
   - Linux: 'NanumGothic' (나눔고딕)

4. 애니메이션 파일 저장 경로 변경
   - 애니메이션 파일이 상위 디렉토리에 저장되도록 수정
   - `charge_log.csv` 입력 시 자동으로 `simulation_log.csv` 사용

## 사용 방법

### 시뮬레이션 실행
```bash
python main.py [옵션]
```

주요 옵션:
- `--time`: 시뮬레이션 시간 (초)
- `--normal`: 일반 차량 수
- `--ev`: 전기차 수
- `--visualize`: 시뮬레이션 결과 시각화
- `--no-save-csv`: CSV 저장 비활성화

예시:
```bash
python main.py --time 7200 --normal 25 --ev 5 --visualize
```

### 애니메이션 생성
```bash
python parking_animation.py <simulation_log.csv> [output.mp4]
```

옵션:
- `--fps`: 애니메이션 프레임 속도 (기본값: 10)
- `--dpi`: 이미지 해상도 (기본값: 100)
- `--speed`: 시뮬레이션 속도 (기본값: 60.0)

예시:
```bash
python parking_animation.py ../results_sim_20240522_123456/simulation_log.csv
```

## 결과 파일
시뮬레이션 결과는 상위 디렉토리의 `results_sim_YYYYMMDD_HHMMSS` 폴더에 저장됩니다:
- `simulation_log.csv`: 시뮬레이션 로그
- `charge_log.csv`: 충전 관련 로그
- `arrivals_by_hour.png`: 시간대별 차량 도착 그래프
- `parking_duration.png`: 주차 시간 분포 그래프
- `charging_patterns.png`: 충전 패턴 그래프
- `parking_animation_[timestamp].mp4`: 주차장 상태 변화 애니메이션

## 시스템 요구사항
- Python 3.8 이상
- matplotlib
- pandas
- numpy
- ffmpeg (애니메이션 저장용, 선택사항)

## 운영체제별 한글 폰트
- Windows: 맑은 고딕 (Malgun Gothic)
- macOS: 애플 고딕 (AppleGothic)
- Linux: 나눔고딕 (NanumGothic)