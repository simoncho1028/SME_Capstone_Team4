"""
주차장 내 차량을 나타내는 모델 클래스
"""
from typing import Tuple, Optional, List
import simpy
import random

from src.utils.helpers import (
    sample_battery_level, sample_parking_duration, 
    sample_charge_time, get_next_coord, cell_type, find_entrance,
    find_nearest_parking_spot, find_all_parking_spots, sample_time_dependent_parking_duration
)
from src.utils.logger import SimulationLogger
from src.config import (
    CELL_ENTRANCE, CELL_PARK, CELL_CHARGER, CELL_ROAD,
    CELL_SIZE_LENGTH, DRIVING_SPEED_MS, PARKING_TIME,
    PARKING_MAP
)

# 주차 가능한 위치 미리 계산
PARKING_SPOTS = find_all_parking_spots()
# 접근 가능한 주차 위치 (실제 도로에서 직접 접근 가능한 위치)
ACCESSIBLE_PARKING_SPOTS = []
# 주차면별 가장 가까운 도로 위치 저장
PARKING_ROAD_ACCESS = {}

# 접근 가능한 주차면 계산 (도로에 인접한 주차면)
for r, c in PARKING_SPOTS:
    # 인접한 셀 중 하나라도 도로이면 접근 가능
    neighbors = [(r-1, c), (r+1, c), (r, c-1), (r, c+1)]
    road_neighbors = []
    
    for nr, nc in neighbors:
        if 0 <= nr < len(PARKING_MAP) and 0 <= nc < len(PARKING_MAP[0]):
            if PARKING_MAP[nr][nc] == CELL_ROAD:
                road_neighbors.append((nr, nc))
                if (r, c) not in ACCESSIBLE_PARKING_SPOTS:
                    ACCESSIBLE_PARKING_SPOTS.append((r, c))
    
    # 접근 가능한 도로 위치 저장
    if road_neighbors:
        PARKING_ROAD_ACCESS[(r, c)] = road_neighbors

# 접근 가능한 주차면이 없으면 모든 주차면 사용
if not ACCESSIBLE_PARKING_SPOTS:
    ACCESSIBLE_PARKING_SPOTS = PARKING_SPOTS

# === 일방통행 경로 하드코딩 (출차 경로) ===
ROAD_PATH = [
    (1,1),(2,1),(3,1),(4,1),(5,1),(6,1),(7,1),(8,1),(9,1), # 아래로
    (9,2),(9,3),(9,4), # 오른쪽
    (8,4),(7,4),(6,4),(5,4),(4,4),(3,4),(2,4),(1,4), # 위로
    (1,3), # 왼쪽
    (0,3) # 출구(입구)
]

class Vehicle:
    """
    주차장 내 차량을 나타내는 클래스
    일반 차량 및 전기차 모두 이 클래스로 표현됩니다.
    """
    
    # 현재 주차된 차량의 위치를 추적
    occupied_spots: List[Tuple[int, int]] = []
    
    def __init__(self, 
                 vid: int, 
                 vtype: str, 
                 env: simpy.Environment,
                 parking_res: simpy.Resource, 
                 charger_res: simpy.Resource,
                 logger: SimulationLogger):
        """
        차량 객체를 초기화합니다.
        
        Args:
            vid: 차량 고유 ID
            vtype: 차량 유형 ("normal" 또는 "ev")
            env: SimPy 환경 객체
            parking_res: 일반 주차면 리소스
            charger_res: EV 충전소 리소스
            logger: 이벤트 로깅을 위한 로거 객체
        """
        self.id = vid
        self.type = vtype
        self.env = env
        self.parking_res = parking_res
        self.charger_res = charger_res
        self.logger = logger
        
        # 전기차만 배터리 속성 초기화 (0-100%)
        # 모든 전기차의 배터리 용량을 100으로 설정, 초기 배터리는 랜덤
        self.battery_capacity = 100.0 if vtype == "ev" else None
        self.battery = sample_battery_level() if vtype == "ev" else None
        
        # 차량 위치 초기화 (입구에서 시작)
        self.pos = find_entrance()
        
        # 사용 중인 리소스 추적
        self.current_resource: Optional[simpy.Resource] = None
        self.resource_request: Optional[simpy.Request] = None
        
        # 주차 시간 초기화
        self.parking_duration = sample_time_dependent_parking_duration(env)
        
        # 차량 도착 이벤트 로깅
        self.log_event("arrive")
        
        # 주차면 탐색 관련 설정
        self.max_search_attempts = 8  # 주차면 최대 탐색 시도 횟수
    
    def log_event(self, event: str) -> None:
        """
        차량 이벤트를 로그에 기록합니다.
        
        Args:
            event: 이벤트 유형
        """
        self.logger.add_event(
            vehicle_id=self.id,
            vehicle_type=self.type,
            event=event,
            time=self.env.now,
            pos=self.pos,
            battery=self.battery
        )
    
    def move_to_position(self, target_r: int, target_c: int):
        """
        도로를 따라 목표 위치로 이동하는 프로세스
        
        Args:
            target_r: 목표 행 위치
            target_c: 목표 열 위치
            
        Yields:
            이동 완료 이벤트
        """
        current_r, current_c = self.pos
        visited_positions = set()  # 이미 방문한 위치 기록
        attempts = 0
        max_movement_attempts = 12  # 최대 이동 시도 횟수
        print(f"[DEBUG] 차량 {self.id} 이동 시작 - 현재: {self.pos}, 목표: ({target_r}, {target_c})")
        while (current_r, current_c) != (target_r, target_c) and attempts < max_movement_attempts:
            # 일방통행: 왼쪽→아래→오른쪽→위 순서로만 이동
            directions = [(0, -1), (1, 0), (0, 1), (-1, 0)]  # 왼, 아래, 오른, 위
            moved = False
            for dr, dc in directions:
                nr, nc = current_r + dr, current_c + dc
                if 0 <= nr < len(PARKING_MAP) and 0 <= nc < len(PARKING_MAP[0]):
                    if PARKING_MAP[nr][nc] in (CELL_ROAD, CELL_ENTRANCE) and (nr, nc) not in visited_positions:
                        next_pos = (nr, nc)
                        moved = True
                        break
            if not moved:
                # 이동할 곳이 없으면 방문 기록 초기화 후 get_next_coord 사용
                visited_positions.clear()
                next_pos = get_next_coord(current_r, current_c)
            visited_positions.add(next_pos)
            current_r, current_c = next_pos
            self.pos = (current_r, current_c)
            self.log_event("move")
            move_time = CELL_SIZE_LENGTH / DRIVING_SPEED_MS
            yield self.env.timeout(move_time)
            if (current_r, current_c) == (target_r, target_c):
                print(f"[DEBUG] 차량 {self.id} 목표 위치에 도달 - 위치: {self.pos}, 이동 횟수: {attempts+1}")
                break
            attempts += 1
            manhattan_distance = abs(current_r - target_r) + abs(current_c - target_c)
            print(f"[DEBUG] 차량 {self.id} 이동 중 - 현재: {self.pos}, 목표까지 거리: {manhattan_distance}, 시도: {attempts}/{max_movement_attempts}")
        if attempts >= max_movement_attempts:
            print(f"[WARN] 차량 {self.id} 이동 시도 횟수 초과 - 현재: {self.pos}, 목표: ({target_r}, {target_c})")
        return
    
    def drive_to_spot(self):
        """
        가장 가까운 빈 주차면을 찾아 이동하는 프로세스
        
        Yields:
            주차 성공 이벤트
        """
        found_spot = False
        attempts = 0
        visited_spots = set()  # 이미 방문한 주차면 추적
        
        # 디버그: 검색 시작 지점 로깅
        print(f"[DEBUG] 차량 {self.id} ({self.type}) 주차면 검색 시작 - 위치: {self.pos}")
        
        # 전기차는 충전소 우선, 일반 차량은 일반 주차면 우선
        want_charger = self.type == "ev" and self.battery < 80
        
        while not found_spot and attempts < self.max_search_attempts:
            r, c = self.pos
            current_cell = cell_type(r, c)
            
            # 현재 위치가 주차 가능한 셀인지 확인
            if current_cell == CELL_PARK:
                # 일반 주차면은 모든 차량이 이용 가능
                try:
                    self.resource_request = self.parking_res.request()
                    yield self.resource_request
                    
                    self.current_resource = self.parking_res
                    # 주차된 위치 추가
                    Vehicle.occupied_spots.append(self.pos)
                    self.log_event("park_start")
                    found_spot = True
                    print(f"[DEBUG] 차량 {self.id} ({self.type}) 주차 성공 - 위치: {self.pos}, 일반 주차면")
                    
                except Exception as e:
                    print(f"[ERROR] 차량 {self.id} 주차 실패: {e}")
                
            elif current_cell == CELL_CHARGER and self.type == "ev" and self.battery < 100:
                # 전기차만 충전소 이용 가능 (배터리 100% 미만)
                try:
                    self.resource_request = self.charger_res.request()
                    yield self.resource_request
                    
                    self.current_resource = self.charger_res
                    # 주차된 위치 추가
                    Vehicle.occupied_spots.append(self.pos)
                    self.log_event("park_start")
                    found_spot = True
                    print(f"[DEBUG] 차량 {self.id} ({self.type}) 주차 성공 - 위치: {self.pos}, 충전소")
                    
                except Exception as e:
                    print(f"[ERROR] 차량 {self.id} 충전소 이용 실패: {e}")
            
            # 주차 실패한 경우 새로운 주차면 찾기 시도
            if not found_spot:
                # 접근 가능한 주차면 목록에서 아직 방문하지 않은 곳 필터링
                available_spots = [spot for spot in ACCESSIBLE_PARKING_SPOTS 
                                  if spot not in Vehicle.occupied_spots and spot not in visited_spots]
                
                # 전기차는 충전소를 먼저 찾도록 함
                if self.type == "ev" and self.battery < 100:
                    # 충전소 위치 필터링
                    charger_spots = [spot for spot in available_spots 
                                   if cell_type(*spot) == CELL_CHARGER]
                    if charger_spots:
                        available_spots = charger_spots
                        print(f"[DEBUG] 차량 {self.id} 충전소 탐색 - 사용 가능한 충전소 수: {len(charger_spots)}")
                
                if not available_spots:
                    # 모든 접근 가능한 주차면을 시도했다면 다시 전체에서 검색
                    available_spots = [spot for spot in PARKING_SPOTS 
                                     if spot not in Vehicle.occupied_spots and spot not in visited_spots]
                
                if not available_spots:
                    # 모든 주차면을 다 시도했다면 다시 방문 기록 초기화 (한 바퀴 다 돌았으므로)
                    visited_spots.clear()
                    available_spots = [spot for spot in ACCESSIBLE_PARKING_SPOTS 
                                      if spot not in Vehicle.occupied_spots]
                
                if not available_spots:
                    # 여전히 없으면 입구로 이동
                    next_pos = find_entrance()
                    print(f"[WARN] 차량 {self.id} 가용 주차면 없음, 입구로 이동")
                    yield from self.move_to_position(next_pos[0], next_pos[1])
                else:
                    # 방문 가능한 주차면 중에서 현재 위치에서 가장 가까운 곳 선택
                    # 단, 일방통행으로 인해 실제 주행 거리 계산은 복잡하므로 맨해튼 거리 사용
                    def manhattan_distance(spot):
                        return abs(spot[0] - r) + abs(spot[1] - c)
                    
                    # 가까운 순서로 정렬 후 최대 3개 중에서 랜덤 선택
                    sorted_spots = sorted(available_spots, key=manhattan_distance)
                    candidates = sorted_spots[:min(3, len(sorted_spots))]
                    target_spot = random.choice(candidates)
                    
                    print(f"[DEBUG] 차량 {self.id} 목표 주차면 선택 - 위치: {target_spot}")
                    
                    # 목표 주차면 방문 기록에 추가
                    visited_spots.add(target_spot)
                    
                    # 주차면에 접근하기 위한 도로 위치 찾기
                    road_access_points = PARKING_ROAD_ACCESS.get(target_spot, [])
                    
                    if road_access_points:
                        # 가장 가까운 도로 접근 지점 선택
                        access_point = min(road_access_points, 
                                         key=lambda p: abs(p[0] - r) + abs(p[1] - c))
                        
                        # 먼저 도로 접근 지점으로 이동
                        print(f"[DEBUG] 차량 {self.id} 주차면 접근 위해 도로 지점으로 이동 - 위치: {access_point}")
                        yield from self.move_to_position(access_point[0], access_point[1])
                        
                        # 그 다음 주차면으로 이동
                        print(f"[DEBUG] 차량 {self.id} 도로에서 주차면으로 이동 시도 - 목표: {target_spot}")
                        self.pos = target_spot
                        yield self.env.timeout(PARKING_TIME / 2)  # 주차 과정 시간의 절반을 주차 이동에 소요
                    else:
                        # 도로 접근 정보가 없으면 직접 이동 시도
                        yield from self.move_to_position(target_spot[0], target_spot[1])
                
                attempts += 1
        
        # 최대 시도 횟수를 초과한 경우
        if not found_spot:
            print(f"[ERROR] 차량 {self.id} ({self.type}) 주차면 찾기 실패 - 최대 시도 횟수 초과")
            # 입구로 이동
            entrance = find_entrance()
            yield from self.move_to_position(entrance[0], entrance[1])
            
            # 강제 주차 (시뮬레이션 계속 진행을 위함)
            self.resource_request = self.parking_res.request()
            yield self.resource_request
            self.current_resource = self.parking_res
            Vehicle.occupied_spots.append(self.pos)
            self.log_event("park_start")
            print(f"[DEBUG] 차량 {self.id} 입구에 강제 주차됨")
    
    def charge(self):
        """
        배터리 충전 프로세스 (전기차만 해당)
        
        Yields:
            충전 완료 이벤트
        """
        if self.type != "ev" or self.battery is None or self.battery >= 100:
            return
        
        self.log_event("charge_start")
        
        # 현재 배터리 레벨 기록
        initial_battery = self.battery
        target_battery = 100.0  # 목표 충전량
        
        # 충전 속도: 5분(300초)에 1% 충전
        charge_rate = 0.162 / 60 # %/초
        
        # 필요한 총 충전 시간 계산
        total_charge_needed = target_battery - initial_battery
        total_charge_time = total_charge_needed / charge_rate  # 초 단위
        
        print(f"[DEBUG] 차량 {self.id} 충전 시작 - 위치: {self.pos}, 배터리: {self.battery}%, 필요 시간: {total_charge_time/60:.1f}분")
        
        # 10% 단위로 충전 상태를 로깅하기 위한 다음 로깅 지점
        next_log_threshold = initial_battery + 10.0
        if next_log_threshold > 100.0:
            next_log_threshold = 100.0
        
        # 실제 충전 과정 시뮬레이션
        charge_step_time = 60.0  # 1분 단위로 충전 상태 업데이트
        remaining_time = total_charge_time
        
        while remaining_time > 0 and self.battery < 100.0:
            # 이번 단계에서 충전할 시간 결정
            step_time = min(charge_step_time, remaining_time)
            
            # 충전 실행
            yield self.env.timeout(step_time)
            
            # 배터리 업데이트
            charge_amount = step_time * charge_rate
            self.battery = min(100.0, self.battery + charge_amount)
            remaining_time -= step_time
            
            # 10% 단위로 충전 상태 로깅
            if self.battery >= next_log_threshold:
                print(f"[DEBUG] 차량 {self.id} 충전 중 - 배터리: {self.battery:.1f}%, 남은 시간: {remaining_time/60:.1f}분")
                # 추가 로깅: 배터리 상태 업데이트
                self.log_event("charge_update")
                # 다음 로깅 지점 설정
                next_log_threshold = min(100.0, next_log_threshold + 10.0)
        
        # 충전 완료
        self.battery = 100.0  # 최종 배터리 상태 100%로 설정
        self.log_event("charge_end")
        print(f"[DEBUG] 차량 {self.id} 충전 완료 - 위치: {self.pos}, 배터리: {self.battery}%")
    
    def depart(self):
        """
        주차 후 일정 시간 대기 후 출차하는 프로세스
        
        Yields:
            출차 이벤트
        """
        # 주차 과정 시간 추가 (30초 고정)
        print(f"[DEBUG] 차량 {self.id} 주차 진행 중 - 위치: {self.pos}, 소요 시간: {PARKING_TIME}초")
        yield self.env.timeout(PARKING_TIME)  # 고정된 주차 소요 시간
        
        # 저장된 주차 시간만큼 대기
        print(f"[DEBUG] 차량 {self.id} 주차 완료 - 위치: {self.pos}, 예상 대기 시간: {self.parking_duration/60:.1f}분")
        yield self.env.timeout(self.parking_duration)
        
        # 리소스 반환 (주차면 또는 충전소)
        if self.current_resource and self.resource_request:
            self.current_resource.release(self.resource_request)
            self.current_resource = None
            self.resource_request = None
            # 주차 위치 제거
            if self.pos in Vehicle.occupied_spots:
                Vehicle.occupied_spots.remove(self.pos)
        self.log_event("depart")
        print(f"[DEBUG] 차량 {self.id} 출차 시작 - 위치: {self.pos}")
        # 정확한 일방통행 경로를 따라 이동
        road_path = ROAD_PATH
        # 현재 위치가 경로상 어디인지 찾기
        try:
            idx = road_path.index(self.pos)
        except ValueError:
            # 경로에 없다면 가장 가까운 경로 셀로 이동
            min_dist = float('inf')
            idx = 0
            for i, (r, c) in enumerate(road_path):
                dist = abs(self.pos[0]-r) + abs(self.pos[1]-c)
                if dist < min_dist:
                    min_dist = dist
                    idx = i
            self.pos = road_path[idx]
            self.log_event("move")
            move_time = CELL_SIZE_LENGTH / DRIVING_SPEED_MS
            yield self.env.timeout(move_time)
        # 경로를 따라 한 칸씩 이동
        while self.pos != (0,3):
            idx = road_path.index(self.pos)
            next_pos = road_path[idx+1]
            self.pos = next_pos
            self.log_event("move")
            move_time = CELL_SIZE_LENGTH / DRIVING_SPEED_MS
            yield self.env.timeout(move_time)
        print(f"[DEBUG] 차량 {self.id} 출차 완료 - 최종 위치: {self.pos}")
    
    def process(self):
        """
        차량의 전체 생애주기 프로세스
        주차→(충전)→출차 과정을 관리합니다.
        
        Yields:
            전체 프로세스 이벤트
        """
        # 1. 주차 과정
        yield from self.drive_to_spot()
        
        # 2. 전기차 충전 (조건 충족 시)
        if self.type == "ev" and self.battery < 100 and cell_type(*self.pos) == CELL_CHARGER:
            yield from self.charge()
        
        # 3. 출차 과정
        yield from self.depart() 