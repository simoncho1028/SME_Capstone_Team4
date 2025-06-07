"""
시뮬레이션에 필요한 유틸리티 함수들을 제공하는 모듈
"""
import random
import numpy as np
from typing import List, Tuple, Dict, Optional

# 시간대별 입차 비율 (24시간)
normalized_entry_ratios = [
    0.013531, 0.006599, 0.002784, 0.001782, 0.001392, 0.003202,
    0.007406, 0.016510, 0.017680, 0.013531, 0.018236, 0.039313,
    0.047053, 0.019823, 0.019267, 0.023387, 0.045438, 0.103878,
    0.184676, 0.133307, 0.113122, 0.084528, 0.055489, 0.028065
]

# 시간대별 주차 시간 감마 분포 파라미터 (shape, scale)
gamma_params_by_hour = [
    (0.496055, 3348.350441),
    (0.969283, 2262.20704),
    (1.129291, 1958.911718),
    (0.895076, 2830.152868),
    (0.818389, 2632.409772),
    (1.037602, 2590.069852),
    (1.018489, 3100.218809),
    (0.914048, 2917.078389),
    (0.818248, 3231.765518),
    (0.838038, 3139.269492),
    (0.849557, 3135.530798),
    (0.901478, 2663.58328),
    (0.726321, 3296.328244),
    (0.885395, 2552.888452),
    (1.094405, 2231.224909),
    (0.905449, 2677.442537),
    (1.085839, 2444.461224),
    (0.945138, 2478.501178),
    (0.992037, 2556.792216),
    (1.027205, 2298.387643),
    (1.038706, 2277.714198),
    (1.076094, 2466.355167),
    (1.015639, 2488.480821),
    (1.165172, 2681.346973)
]

EV_CHARGING_MEAN = 2.0          # EV 평균 충전 시간 (시간)
EV_CHARGING_STD = 1.0           # EV 충전 시간 표준편차 (시간)

def sample_battery_level() -> float:
    """
    전기차의 초기 배터리 잔량을 샘플링합니다.
    
    Returns:
        float: 0-100 사이의 배터리 잔량
    """
    # 평균 40%, 표준편차 20%의 정규분포에서 샘플링
    battery = np.random.normal(50, 20)
    # 0-100 범위로 제한
    return max(0, min(100, battery))

def sample_parking_duration(arrival_time: float) -> float:
    """
    차량의 주차 시간을 시간대별 감마분포(under24/over24)에서 샘플링합니다.
    - 27% 확률로 over24, 73% 확률로 under24
    - arrival_time(초) 기준으로 시간대(hour) 추출 (0~23)
    - 각 분포의 shape, scale 파라미터 적용
    Returns:
        float: 주차 시간 (초)
    """
    # 시간대(hour) 추출 (0~23)
    hour = int((arrival_time % 86400) // 3600)

    # 확률로 분포 선택
    if random.random() < 0.27:
        # over24
        shape, scale = [
            (2.842, 1668.003), (4.697, 870.481), (4.933, 749.986), (2.493, 1596.151),
            (3.698, 1054.659), (5.57, 668.405), (3.824, 1041.937), (4.225, 989.981),
            (3.69, 1122.099), (4.602, 975.278), (4.613, 973.55), (4.013, 949.655),
            (4.692, 947.418), (5.074, 739.922), (6.094, 635.911), (4.467, 945.921),
            (4.8, 892.503), (5.606, 834.248), (4.951, 926.108), (4.975, 902.71),
            (4.732, 939.169), (4.605, 975.659), (4.358, 1005.49), (3.573, 1207.395)
        ][hour]
        minutes = np.random.gamma(shape, scale)
        duration = minutes * 60  # 시간 제한 없음
    else:
        # under24
        shape, scale = [
            (6.974, 76.528), (4.263, 104.897), (2.904, 119.591), (1.002, 238.622),
            (0.409, 379.759), (0.429, 240.593), (0.686, 222.459), (1.142, 240.409),
            (0.557, 382.151), (0.681, 525.304), (0.597, 565.776), (0.369, 500.576),
            (0.287, 608.889), (0.429, 633.819), (0.951, 483.534), (0.977, 495.728),
            (1.633, 355.543), (2.116, 275.197), (2.233, 247.028), (3.948, 160.379),
            (5.834, 108.328), (5.821, 101.914), (5.8, 96.547), (8.808, 62.592)
        ][hour]
        minutes = np.random.gamma(shape, scale)
        duration = minutes * 60  # 시간 제한 없음
    return duration

def sample_charge_time() -> float:
    """
    전기차 충전 시간을 샘플링합니다.
    
    Returns:
        float: 충전 시간 (초)
    """
    # 평균 2시간, 표준편차 1시간의 정규분포에서 샘플링
    hours = np.random.normal(2, 1)
    # 최소 30분, 최대 4시간으로 제한
    hours = max(0.5, min(4, hours))
    return hours * 3600  # 초 단위로 변환

def sample_interarrival_time() -> float:
    """
    차량 간 도착 시간 간격을 샘플링합니다.
    
    Returns:
        float: 도착 시간 간격 (초)
    """
    # 평균 5분의 지수분포에서 샘플링
    minutes = np.random.exponential(5)
    # 최소 1분, 최대 15분으로 제한
    minutes = max(1, min(15, minutes))
    return minutes * 60  # 초 단위로 변환

def calculate_manhattan_distance(pos1: Tuple[int, int], pos2: Tuple[int, int]) -> int:
    """
    두 위치 간의 맨해튼 거리를 계산합니다.
    
    Args:
        pos1: 첫 번째 위치 (행, 열)
        pos2: 두 번째 위치 (행, 열)
        
    Returns:
        int: 맨해튼 거리
    """
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

def calculate_parking_time(start_pos: Tuple[int, int], end_pos: Tuple[int, int]) -> float:
    """
    주차에 걸리는 시간을 계산합니다.
    
    Args:
        start_pos: 시작 위치 (행, 열)
        end_pos: 목표 위치 (행, 열)
        
    Returns:
        float: 주차 시간 (초)
    """
    # 맨해튼 거리당 3초로 계산
    distance = calculate_manhattan_distance(start_pos, end_pos)
    return distance * 3

def find_nearest_available_spot(
    current_pos: Tuple[int, int],
    occupied_spots: List[Tuple[int, int]],
    spot_type: str
) -> Optional[Tuple[int, int]]:
    """
    현재 위치에서 가장 가까운 비어있는 주차면을 찾습니다.
    
    Args:
        current_pos: 현재 위치 (행, 열)
        occupied_spots: 이미 점유된 주차면 위치 목록
        spot_type: 찾을 주차면 타입 ("P": 일반, "C": 충전소)
        
    Returns:
        Optional[Tuple[int, int]]: 가장 가까운 비어있는 주차면의 위치, 없으면 None
    """
    # 임시로 랜덤한 위치 반환 (실제 구현에서는 주차장 레이아웃에 따라 수정 필요)
    while True:
        spot = (random.randint(1, 10), random.randint(1, 10))
        if spot not in occupied_spots:
            return spot 