"""
주차장 시뮬레이션에 필요한 유틸리티 함수들을 제공하는 모듈입니다.
"""
import random
import numpy as np
from typing import Tuple, List, Optional, Callable

from src.config import PARKING_MAP, CELL_ENTRANCE, CELL_ROAD


def cell_type(r: int, c: int) -> str:
    """
    주차장 맵에서 (r,c) 셀 타입을 반환합니다.
    
    Args:
        r: 행 인덱스
        c: 열 인덱스
        
    Returns:
        셀의 타입 문자 ('E', 'R', 'P', 'C', 'N' 중 하나)
    """
    if 0 <= r < len(PARKING_MAP) and 0 <= c < len(PARKING_MAP[0]):
        return PARKING_MAP[r][c]
    return "N"  # 맵 범위 밖은 경계로 처리


def get_next_coord(r: int, c: int) -> Tuple[int, int]:
    """
    반시계 일방통행: 왼→아래→오른→위 순서로 다음 R/E 셀 좌표를 반환합니다.
    
    Args:
        r: 현재 행 위치
        c: 현재 열 위치
        
    Returns:
        다음 이동할 셀의 (행, 열) 좌표
    """
    # 반시계 방향 순서로 이동 방향 정의
    directions = [(0, -1), (1, 0), (0, 1), (-1, 0)]  # 왼, 아래, 오른, 위
    
    # 모든 방향 시도
    for dr, dc in directions:
        nr, nc = r + dr, c + dc
        if 0 <= nr < len(PARKING_MAP) and 0 <= nc < len(PARKING_MAP[0]):
            if PARKING_MAP[nr][nc] in (CELL_ROAD, CELL_ENTRANCE):
                return nr, nc
    
    # 인접한 셀에 도로나 입구가 없으면 더 넓은 범위 탐색
    for distance in range(2, 4):
        for dr, dc in directions:
            nr, nc = r + dr * distance, c + dc * distance
            if 0 <= nr < len(PARKING_MAP) and 0 <= nc < len(PARKING_MAP[0]):
                # 목표 셀이 도로나 입구인 경우, 해당 방향으로 1칸씩 이동
                if PARKING_MAP[nr][nc] in (CELL_ROAD, CELL_ENTRANCE):
                    # 첫 번째 단계 이동
                    nr1, nc1 = r + dr, c + dc
                    if 0 <= nr1 < len(PARKING_MAP) and 0 <= nc1 < len(PARKING_MAP[0]):
                        return nr1, nc1
    
    # 맵 중앙 쪽으로 이동 시도
    center_r, center_c = len(PARKING_MAP) // 2, len(PARKING_MAP[0]) // 2
    if r < center_r:
        nr, nc = r + 1, c  # 아래로
    elif r > center_r:
        nr, nc = r - 1, c  # 위로
    elif c < center_c:
        nr, nc = r, c + 1  # 오른쪽으로
    elif c > center_c:
        nr, nc = r, c - 1  # 왼쪽으로
    else:
        # 중앙에 있는 경우 랜덤 방향
        dr, dc = random.choice(directions)
        nr, nc = r + dr, c + dc
    
    # 좌표가 유효한지 확인
    if 0 <= nr < len(PARKING_MAP) and 0 <= nc < len(PARKING_MAP[0]):
        return nr, nc
    
    # 모든 시도가 실패하면 입구로 이동
    entrance = find_entrance()
    if entrance != (r, c):
        return entrance
    
    # 마지막 대안으로 제자리 반환
    return r, c


def find_entrance() -> Tuple[int, int]:
    """
    맵에서 입구/출구('E') 좌표를 검색합니다.
    
    Returns:
        입구/출구 셀의 (행, 열) 좌표
        
    Raises:
        ValueError: 입구/출구가 맵에 없을 경우
    """
    for i, row in enumerate(PARKING_MAP):
        j = row.find(CELL_ENTRANCE)
        if j >= 0:
            return (i, j)
    
    raise ValueError("입구/출구(E)를 찾을 수 없습니다.")


def find_all_parking_spots() -> List[Tuple[int, int]]:
    """
    맵에서 모든 주차 공간('P', 'C')의 좌표를 찾습니다.
    
    Returns:
        모든 주차 공간의 좌표 목록
    """
    spots = []
    for r, row in enumerate(PARKING_MAP):
        for c, cell in enumerate(row):
            if cell in ('P', 'C'):
                spots.append((r, c))
    return spots


def find_nearest_parking_spot(current_r: int, current_c: int, 
                              want_charger: bool = False) -> Optional[Tuple[int, int]]:
    """
    현재 위치에서 가장 가까운 주차 공간을 찾습니다.
    
    Args:
        current_r: 현재 행 위치
        current_c: 현재 열 위치
        want_charger: 충전소 우선 검색 여부
        
    Returns:
        가장 가까운 주차 공간 좌표 또는 None
    """
    # 타겟 셀 유형
    target_types = ['C', 'P'] if want_charger else ['P', 'C']
    
    # 모든 주차 공간 검색
    min_distance = float('inf')
    nearest_spot = None
    
    for target_type in target_types:
        for r, row in enumerate(PARKING_MAP):
            for c, cell in enumerate(row):
                if cell == target_type:
                    # 맨해튼 거리 계산
                    distance = abs(r - current_r) + abs(c - current_c)
                    if distance < min_distance:
                        min_distance = distance
                        nearest_spot = (r, c)
        
        # 해당 타입의 주차 공간을 찾았으면 바로 반환
        if nearest_spot and (want_charger and target_type == 'C'):
            return nearest_spot
    
    return nearest_spot


# 시간대별 입차 간격 분포 파라미터 (λ)
exp_lambdas = [
    0.2356, 0.4831, 1.145, 1.789, 2.290, 0.9957,
    0.4305, 0.1931, 0.1803, 0.2356, 0.1748, 0.0811,
    0.0678, 0.1608, 0.1655, 0.1363, 0.0702, 0.0307,
    0.0173, 0.0239, 0.0282, 0.0377, 0.0575, 0.1136
]

# 하루 총 입차 차량 기준 시간대별 정규화된 입차 비율
normalized_entry_ratios = [
    0.013531, 0.006599, 0.002784, 0.001782, 0.001392, 0.003202,
    0.007406, 0.016510, 0.017680, 0.013531, 0.018236, 0.039313,
    0.047053, 0.019823, 0.019267, 0.023387, 0.045438, 0.103878,
    0.184676, 0.133307, 0.113122, 0.084528, 0.055489, 0.028065
]

# 시간대별 주차 시간 감마 분포 파라미터 (shape, scale)
gamma_params = [
    (2.38, 1024.38), (2.45, 918.24), (1.97, 994.74), (1.68, 1109.27),
    (3.53, 698.02), (3.13, 740.71), (3.81, 630.01), (3.23, 781.09),
    (0.25, 2478.54), (0.89, 1628.65), (0.72, 1660.96), (0.17, 1696.04),
    (0.16, 1776.03), (0.30, 1491.06), (0.96, 924.99), (1.35, 803.32),
    (1.84, 611.51), (1.90, 590.75), (1.85, 516.52), (2.21, 405.02),
    (2.59, 304.79), (2.66, 249.52), (2.53, 201.76), (2.45, 147.55)
]

def sample_time_dependent_interarrival(env) -> float:
    """
    현재 시간대에 따른 입차 간격을 샘플링합니다.
    (이 함수는 더 이상 사용되지 않음)
    """
    current_hour = int(env.now // 3600) % 24
    # 경고: 이 λ 값은 시간당 평균 입차 횟수를 나타내므로, 초당으로 변환하여 사용해야 함.
    # 현재는 generate_realistic_entry_times에서 전체 차량 수 기준으로 샘플링하므로 이 함수는 사용하지 않음.
    lambda_value = exp_lambdas[current_hour]
    # 올바르게 사용하려면 lambda_sec = lambda_value / 3600.0 로 변환 필요
    # random.expovariate(lambda_sec)
    return float('inf') # 이 함수는 사용하지 않도록 무한대 반환

def sample_time_dependent_parking_duration(env) -> float:
    """
    현재 시간대에 따른 주차 시간을 샘플링합니다.
    감마 분포 파라미터(scale은 분 단위)를 사용하며, 결과를 초 단위로 반환합니다.
    
    Args:
        env: SimPy 환경 객체
        
    Returns:
        float: 주차 시간 (초)
    """
    current_hour = int(env.now // 3600) % 24
    shape, scale_minutes = gamma_params[current_hour]
    
    # 감마 분포에서 분 단위로 샘플링
    parking_duration_minutes = np.random.gamma(shape, scale_minutes)
    
    # 초 단위로 변환하여 반환
    return parking_duration_minutes * 60.0

# 기존 함수들은 기본값으로 유지 (호환성을 위해 남겨둠)
def sample_interarrival() -> float:
    """
    다음 차량 도착까지의 시간 간격을 샘플링합니다.
    평균 300초(5분)의 지수 분포 사용
    """
    return random.expovariate(1/300)

def sample_parking_duration() -> float:
    """
    주차 지속 시간을 샘플링합니다.
    평균 3600초(1시간)의 지수 분포 사용
    """
    return random.expovariate(1/3600)

def sample_battery_level() -> float:
    """
    전기차의 초기 배터리 잔량을 샘플링합니다.
    충전이 필요하도록 0~50% 범위의 균등 분포 사용
    """
    return random.uniform(0, 50)

def sample_charge_time() -> float:
    """
    충전에 소요되는 시간을 샘플링합니다.
    0~5시간(18000초) 범위의 균등 분포 사용
    """
    return random.uniform(0, 5*3600) 