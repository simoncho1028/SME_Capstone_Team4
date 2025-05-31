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
CELL_SIZE_LENGTH = 5.0  # 셀의 길이 (m)
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

def generate_adjacent_charger_layouts(base_map: List[str]) -> List[List[str]]:
    """
    인접한 2개의 충전소에 대한 모든 가능한 배치 조합을 생성합니다.
    
    Args:
        base_map: 기본 주차장 맵 (충전소 없는 상태)
    
    Returns:
        가능한 모든 충전소 배치의 리스트
    """
    # 충전소 설치 가능한 위치 찾기 (일반 주차면 위치)
    possible_spots = []
    for i, row in enumerate(base_map):
        for j, cell in enumerate(row):
            if cell == 'P':
                possible_spots.append((i, j))
    
    # 인접한 위치 쌍 찾기
    adjacent_pairs = []
    for i, (r1, c1) in enumerate(possible_spots):
        for r2, c2 in possible_spots[i+1:]:
            # 상하좌우로 인접한 경우만 포함
            if (abs(r1 - r2) == 1 and c1 == c2) or (r1 == r2 and abs(c1 - c2) == 1):
                adjacent_pairs.append(((r1, c1), (r2, c2)))
    
    # 각 인접 쌍에 대해 새로운 레이아웃 생성
    layouts = []
    for pair in adjacent_pairs:
        new_layout = [list(row) for row in base_map]
        for r, c in pair:
            new_layout[r][c] = 'C'
        layouts.append([''.join(row) for row in new_layout])
    
    return layouts 