"""
시뮬레이션에 필요한 유틸리티 함수들을 제공하는 모듈
"""
import random
import numpy as np
from typing import List, Tuple, Dict, Optional
from src.config import MIN_PARKING_TIME, MAX_PARKING_TIME, MIN_CHARGING_TIME, MAX_CHARGING_TIME

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

NORMAL_PARKING_MEAN = 4.0       # 일반 차량 평균 주차 시간 (시간)
NORMAL_PARKING_STD = 2.0        # 일반 차량 주차 시간 표준편차 (시간)
EV_CHARGING_MEAN = 2.0          # EV 평균 충전 시간 (시간)
EV_CHARGING_STD = 1.0           # EV 충전 시간 표준편차 (시간)

def sample_battery_level() -> float:
    """
    전기차의 초기 배터리 잔량을 샘플링합니다.
    
    Returns:
        float: 0-100 사이의 배터리 잔량
    """
    # 평균 40%, 표준편차 20%의 정규분포에서 샘플링
    battery = np.random.normal(40, 20)
    # 0-100 범위로 제한
    return max(0, min(100, battery))

def sample_parking_duration(arrival_time: float) -> float:
    """
    차량의 주차 시간을 정규분포에서 샘플링합니다.
    
    Args:
        arrival_time: 차량 도착 시각 (초)
        
    Returns:
        float: 주차 시간 (초)
    """
    # 새로운 정규분포 기반 샘플링
    # MIN_PARKING_TIME ~ MAX_PARKING_TIME 범위에서 정규분포 샘플링
    min_hours = MIN_PARKING_TIME / 3600  # 초를 시간으로 변환
    max_hours = MAX_PARKING_TIME / 3600  # 초를 시간으로 변환
    
    # 평균을 min과 max의 중점으로, 표준편차를 범위의 1/6로 설정
    # (정규분포에서 99.7%가 평균 ± 3σ 범위에 들어오므로)
    mean_hours = (min_hours + max_hours) / 2
    std_hours = (max_hours - min_hours) / 6
    
    # 정규분포에서 샘플링
    hours = np.random.normal(mean_hours, std_hours)
    
    # 초 단위로 변환하고 제한 적용
    duration = hours * 3600
    duration = max(MIN_PARKING_TIME, min(MAX_PARKING_TIME, duration))
    
    return duration
    
    # 기존 감마 분포 기반 코드 (주석 처리)
    # # 도착 시각을 시간대(0-23)로 변환
    # hour = int((arrival_time % 86400) // 3600)
    # 
    # # 해당 시간대의 감마 분포 파라미터 가져오기
    # shape, scale = gamma_params_by_hour[hour]
    # 
    # # 감마 분포에서 샘플링 (분 단위)
    # minutes = np.random.gamma(shape, scale)
    # 
    # # 초 단위로 변환하고 제한 적용
    # duration = minutes * 60
    # duration = max(MIN_PARKING_TIME, min(MAX_PARKING_TIME, duration))
    # 
    # return duration

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