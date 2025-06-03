"""
시뮬레이션 환경 설정과 관련된 모든 상수 및 구성 값을 관리하는 모듈입니다.
"""
from typing import Dict, List
from src.utils.parking_map_loader import ParkingMapLoader

# 시뮬레이션 기본 설정
SEED = 42                  # 난수 생성기 시드
SIM_TIME = 86_400          # 24시간 (초 단위)

# 차량 설정
NUM_NORMAL = 830           # 일반 차량 수
NUM_EV = 36               # 전기차 수
TOTAL_VEHICLES = NUM_NORMAL + NUM_EV  # 총 차량 수

# 주차장 설정
TOTAL_PARKING_SPOTS = 866  # 총 주차면 수
DEFAULT_CHARGER_COUNT = NUM_EV  # EV 충전소 수 (전기차 수와 동일하게 설정)
DEFAULT_BUILDING_COUNT = 8  # 건물 동 수

# 시간 설정 (초 단위)
MIN_PARKING_TIME = 30 * 60      # 최소 주차 시간 (30분)
MAX_PARKING_TIME = 12 * 3600    # 최대 주차 시간 (12시간)
MIN_CHARGING_TIME = 30 * 60     # 최소 충전 시간 (30분)
MAX_CHARGING_TIME = 4 * 3600    # 최대 충전 시간 (4시간)
MIN_ARRIVAL_INTERVAL = 60       # 최소 도착 간격 (1분)
MAX_ARRIVAL_INTERVAL = 15 * 60  # 최대 도착 간격 (15분)

# 배터리 설정
DEFAULT_BATTERY_MEAN = 40.0     # 평균 초기 배터리 잔량 (%)
DEFAULT_BATTERY_STD = 20.0      # 배터리 잔량 표준편차 (%)
MIN_BATTERY_LEVEL = 0.0         # 최소 배터리 잔량 (%)
MAX_BATTERY_LEVEL = 100.0       # 최대 배터리 잔량 (%)

# 주차 시간 설정
NORMAL_PARKING_MEAN = 4.0       # 일반 차량 평균 주차 시간 (시간)
NORMAL_PARKING_STD = 2.0        # 일반 차량 주차 시간 표준편차 (시간)
EV_CHARGING_MEAN = 2.0          # EV 평균 충전 시간 (시간)
EV_CHARGING_STD = 1.0           # EV 충전 시간 표준편차 (시간)

# 도착 간격 설정
ARRIVAL_MEAN = 5.0              # 평균 도착 간격 (분)

# 이동 시간 설정
MOVE_TIME_PER_CELL = 30.0       # 셀당 이동 시간 (초)

# 물리적 속성 설정
CELL_SIZE_LENGTH = 5.0          # 셀의 길이 (m)
CELL_SIZE_WIDTH = 2.0           # 셀의 너비 (m)
VEHICLE_LENGTH = 5.0            # 차량 길이 (m)
VEHICLE_WIDTH = 2.0             # 차량 너비 (m)
DRIVING_SPEED = 5.0             # 주행 속도 (km/h)
DRIVING_SPEED_MS = DRIVING_SPEED * 1000 / 3600  # 주행 속도 (m/s)
PARKING_TIME = 30.0             # 주차 소요 시간 (초)

# 주차장 맵 로더 초기화 및 맵 로드
_map_loader = ParkingMapLoader()
PARKING_MAPS = _map_loader.load_all_maps()

# 셀 타입 정의
CELL_ENTRANCE = "E"  # 입구/출구
CELL_ROAD = "R"      # 도로
CELL_PARK = "P"      # 일반 주차면
CELL_CHARGER = "C"   # EV 충전기
CELL_UNUSED = "N"    # 사용하지 않는 공간
CELL_BUILDING = "B"  # 건물
CELL_EXIT = "X"      # 출구

# 층별 주차장 맵 접근을 위한 함수
def get_floor_map(floor: str) -> List[List[str]]:
    """
    특정 층의 주차장 맵 반환
    
    Args:
        floor: 층 식별자 (GF, B1F, B2F, B3F)
        
    Returns:
        List[List[str]]: 해당 층의 주차장 맵
    """
    return PARKING_MAPS.get(floor, [])

# 모든 층의 주차장 맵 반환
def get_all_floor_maps() -> Dict[str, List[List[str]]]:
    """
    모든 층의 주차장 맵 반환
    
    Returns:
        Dict[str, List[List[str]]]: 모든 층의 주차장 맵
    """
    return PARKING_MAPS

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