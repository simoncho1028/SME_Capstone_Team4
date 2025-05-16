"""
주차장 시뮬레이션에 필요한 유틸리티 함수들을 제공하는 모듈입니다.
"""
import random
from typing import Tuple, List, Optional

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


# 분포 정의 함수들 (모두 초 단위 반환)
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