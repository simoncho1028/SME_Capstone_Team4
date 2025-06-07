"""
시뮬레이션 환경 설정과 관련된 모든 상수 및 구성 값을 관리하는 모듈입니다.
"""
from typing import Dict, List
from src.utils.parking_map_loader import ParkingMapLoader

# 시뮬레이션 기본 설정
SEED = 422                  # 난수 생성기 시드
SIM_TIME = 86_400          # 24시간 (초 단위)

# 차량 설정
NUM_NORMAL = 830           # 일반 차량 수
NUM_EV = 36               # 전기차 수
TOTAL_VEHICLES = NUM_NORMAL + NUM_EV  # 총 차량 수
ENTRY_RATIO = 0.9         # 전체 차량 중 입차하는 비율

# 시간대별 입차 비율 (24시간)
normalized_entry_ratios = [
    0.013531, 0.006599, 0.002784, 0.001782, 0.001392, 0.003202,
    0.007406, 0.016510, 0.017680, 0.013531, 0.018236, 0.039313,
    0.047053, 0.019823, 0.019267, 0.023387, 0.045438, 0.103878,
    0.184676, 0.133307, 0.113122, 0.084528, 0.055489, 0.028065
]

# 시간대별 누적 확률 계산
CUMULATIVE_ENTRY_PROBS = []
cumsum = 0.0
for ratio in normalized_entry_ratios:
    cumsum += ratio
    CUMULATIVE_ENTRY_PROBS.append(cumsum)

def get_arrival_time() -> float:
    """
    시간대별 입차 비율에 따라 차량의 도착 시간을 생성합니다.
    
    Returns:
        float: 도착 시간 (초)
    """
    import random
    r = random.random()
    
    # 이진 탐색으로 시간대 찾기
    hour = 0
    left, right = 0, len(CUMULATIVE_ENTRY_PROBS)
    while left < right:
        mid = (left + right) // 2
        if CUMULATIVE_ENTRY_PROBS[mid] < r:
            left = mid + 1
        else:
            right = mid
    hour = left
    
    # 선택된 시간대 내에서 랜덤한 시간 선택
    minute = random.uniform(0, 60)
    return (hour * 3600) + (minute * 60)

# 주차장 설정
TOTAL_PARKING_SPOTS = 866  # 총 주차면 수
DEFAULT_CHARGER_COUNT = NUM_EV  # EV 충전소 수 (전기차 수와 동일하게 설정)
DEFAULT_BUILDING_COUNT = 8  # 건물 동 수

# 시간 설정 (초 단위)
MIN_ARRIVAL_INTERVAL = 60       # 최소 도착 간격 (1분)
MAX_ARRIVAL_INTERVAL = 60 * 60  # 최대 도착 간격 (1시간)

# 배터리 설정
DEFAULT_BATTERY_MEAN = 40.0     # 평균 초기 배터리 잔량 (%)
DEFAULT_BATTERY_STD = 20.0      # 배터리 잔량 표준편차 (%)
MIN_BATTERY_LEVEL = 0.0         # 최소 배터리 잔량 (%)
MAX_BATTERY_LEVEL = 100.0       # 최대 배터리 잔량 (%)

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