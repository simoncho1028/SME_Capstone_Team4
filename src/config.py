"""
시뮬레이션 환경 설정과 관련된 모든 상수 및 구성 값을 관리하는 모듈입니다.
"""
from typing import List

# 시뮬레이션 기본 설정
SEED = 42
SIM_TIME = 86_400  # 24시간 (초 단위)

# 차량 설정
NUM_NORMAL = 25  # 일반 차량 수
NUM_EV = 5       # 전기차 수
TOTAL_SPOTS = 32  # 총 주차 공간 (일반 28 + 충전소 4)

# 물리적 속성 설정
CELL_SIZE_LENGTH = 5.0 `` # 셀의 길이 (m)
CELL_SIZE_WIDTH = 2.0   # 셀의 너비 (m)
VEHICLE_LENGTH = 5.0    # 차량 길이 (m)
VEHICLE_WIDTH = 2.0     # 차량 너비 (m)
DRIVING_SPEED = 5.0     # 주행 속도 (km/h)
DRIVING_SPEED_MS = DRIVING_SPEED * 1000 / 3600  # 주행 속도 (m/s) = 약 1.39m/s
PARKING_TIME = 30.0     # 주차 소요 시간 (초)

# 지도 정의 (10×6 그리드)
# N = 경계/미사용, E = 입구/출구, R = 도로, P = 일반 주차면, C = EV 충전소
PARKING_MAP: List[str] = [
    "NNNENN",
    "PRRRRP",
    "PRPPRP",
    "PRPPRP",
    "PRPPRP",
    "PRPPRP",
    "PRPPRC",
    "PRPPRC",
    "PRPPRC",
    "PRRRRC",
]

# 셀 타입 정의
CELL_ENTRANCE = "E"  # 입구/출구
CELL_ROAD = "R"      # 도로
CELL_PARK = "P"      # 일반 주차면
CELL_CHARGER = "C"   # EV 충전기
CELL_UNUSED = "N"    # 사용하지 않는 공간 