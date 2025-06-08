import random
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

from src.models.vehicle import Vehicle
from src.config import PARKING_MAPS, CELL_PARK, CELL_CHARGER
from parking_system import ParkingSystem  # ParkingSystem import 추가
from src.utils.helpers import sample_battery_level

class ParkingManager:
    """주차장 관리자 클래스"""
    
    def __init__(self):
        """주차장 관리자 초기화"""
        self.parked_vehicles = {}  # vehicle_id -> (floor, row, col)
        self.double_parked = set()  # 이중주차된 차량 ID 집합
        self.parking_spots = {}  # (floor, row, col) -> vehicle_id
        self.ev_chargers = set()  # 충전소 위치 집합
        self.normal_spots = set()  # 일반구역 위치 집합 추가
        
        # 층별 주차면 관리
        self.available_spots_by_floor = {}  # floor -> [(floor, row, col), ...]
        self.available_chargers_by_floor = {}  # floor -> [(floor, row, col), ...]
        
        # 전체 사용 가능한 주차면과 충전소
        self.available_spots = []  # [(floor, row, col), ...]
        self.available_chargers = []  # [(floor, row, col), ...]
        
        # 주차장 맵에서 사용 가능한 주차면 찾기
        self.total_parking_spots = 0  # 전체 주차면 수 (충전소 제외)
        self.total_charger_spots = 0  # 전체 충전소 수
        
        # SimPy 환경 추가
        self.env = None
        
        # 로거 추가
        self.logger = None
        
        # ParkingSystem 인스턴스 추가
        self.parking_system = ParkingSystem()
        
        # 주차장 맵 초기화
        self._initialize_parking_map()
        
        # 층 이름 매핑 추가
        self.floor_mapping = {
            'GF': 'Ground',
            'B1F': 'B1',
            'B2F': 'B2',
            'B3F': 'B3'
        }
        self.reverse_floor_mapping = {v: k for k, v in self.floor_mapping.items()}
    
    def _initialize_parking_map(self):
        """주차장 맵 초기화"""
        # 기존 데이터 초기화
        self.available_spots = []
        self.available_chargers = []
        self.available_spots_by_floor = {}
        self.available_chargers_by_floor = {}
        self.normal_spots = set()
        self.ev_chargers = set()
        self.total_parking_spots = 0
        self.total_charger_spots = 0
        
        # 주차장 맵 초기화
        for floor, floor_map in PARKING_MAPS.items():
            self.available_spots_by_floor[floor] = []
            self.available_chargers_by_floor[floor] = []
            
            for row, line in enumerate(floor_map):
                for col, cell in enumerate(line):
                    spot = (floor, row, col)
                    if cell == CELL_PARK:  # 일반 주차면
                        self.available_spots.append(spot)
                        self.available_spots_by_floor[floor].append(spot)
                        self.normal_spots.add(spot)
                        self.total_parking_spots += 1
                    elif cell == CELL_CHARGER:  # 충전소
                        self.available_chargers.append(spot)
                        self.available_chargers_by_floor[floor].append(spot)
                        self.ev_chargers.add(spot)
                        self.total_charger_spots += 1
        
        print(f"[DEBUG] 초기화된 주차면 수: {self.total_parking_spots}")
        print(f"[DEBUG] 초기화된 충전소 수: {self.total_charger_spots}")
    
    def set_env(self, env):
        """SimPy 환경 설정"""
        self.env = env
    
    def set_logger(self, logger):
        """로거 설정"""
        self.logger = logger
    
    def is_spot_available(self, spot: tuple) -> bool:
        """주차면이 사용 가능한지 확인"""
        return spot not in self.parking_spots
    
    def get_available_spots_count(self) -> int:
        """사용 가능한 주차면 수 반환"""
        return len([spot for spot in self.available_spots if self.is_spot_available(spot)])
    
    def get_available_chargers_count(self) -> int:
        """사용 가능한 충전소 수 반환"""
        return len([spot for spot in self.available_chargers if self.is_spot_available(spot)])
    
    def find_parking_spot(self) -> Optional[Tuple[str, int, int]]:
        """
        사용 가능한 일반 주차면을 찾습니다.
        
        Returns:
            Optional[Tuple[str, int, int]]: (floor, row, column) 또는 None
        """
        # 현재 사용 중인 주차면 제외
        available_spots = []
        for spot in self.available_spots:
            if spot not in self.parking_spots and spot in self.normal_spots:
                available_spots.append(spot)
        
        if not available_spots:
            return None
        return random.choice(available_spots)
    
    def find_ev_charger(self) -> Optional[Tuple[str, int, int]]:
        """
        사용 가능한 EV 충전기를 찾습니다.
        
        Returns:
            Optional[Tuple[str, int, int]]: (floor, row, column) 또는 None
        """
        # 현재 사용 중인 충전기 제외
        available_chargers = []
        for spot in self.available_chargers:
            if spot not in self.parking_spots and spot in self.ev_chargers:
                available_chargers.append(spot)
        
        if not available_chargers:
            return None
        return random.choice(available_chargers)
    
    def get_occupied_normal_count(self) -> int:
        """일반 주차면에 주차된 차량 수 반환"""
        return sum(1 for v in self.parked_vehicles.values() if v not in self.ev_chargers)

    def park_vehicle(self, vehicle: Vehicle) -> Tuple[bool, bool]:
        """
        차량을 주차합니다.
        
        Args:
            vehicle: 주차할 차량
            
        Returns:
            Tuple[bool, bool]: (주차 성공 여부, 충전 실패 여부)
        """
        # 주차 가능한 위치 찾기
        if vehicle.vehicle_type == "ev" and vehicle.needs_charging():
            # EV이고 충전이 필요한 경우 충전소 찾기
            position = self.find_ev_charger()
            if position:
                self.parked_vehicles[vehicle.vehicle_id] = position
                self.parking_spots[position] = vehicle.vehicle_id
                if position in self.available_chargers:
                    self.available_chargers.remove(position)
                return True, False
            else:
                # 충전소가 없으면 일반 주차면 찾기
                position = self.find_parking_spot()
                if position:
                    self.parked_vehicles[vehicle.vehicle_id] = position
                    self.parking_spots[position] = vehicle.vehicle_id
                    if position in self.available_spots:
                        self.available_spots.remove(position)
                    return True, True
                return False, True
        else:
            # 일반 차량이거나 충전이 필요 없는 EV는 일반 주차면 찾기
            position = self.find_parking_spot()
            if position:
                self.parked_vehicles[vehicle.vehicle_id] = position
                self.parking_spots[position] = vehicle.vehicle_id
                if position in self.available_spots:
                    self.available_spots.remove(position)
                return True, False
            return False, False

    def exit_vehicle(self, vehicle: Vehicle) -> bool:
        """
        차량을 출차 처리합니다.
        
        Args:
            vehicle: 출차할 차량
            
        Returns:
            bool: 출차 성공 여부
        """
        # 이중주차된 차량 처리
        if vehicle.vehicle_id in self.double_parked:
            self.double_parked.remove(vehicle.vehicle_id)
            vehicle.update_state("outside")
            # 전기차인 경우 배터리 초기화
            if vehicle.vehicle_type == "ev":
                vehicle.battery_level = sample_battery_level()
            return True
        
        # 일반 주차된 차량 처리
        spot = self.parked_vehicles.get(vehicle.vehicle_id)
        if spot:
            # 주차면에서 차량 정보 제거
            del self.parked_vehicles[vehicle.vehicle_id]
            if spot in self.parking_spots:
                del self.parking_spots[spot]
            
            # 사용 가능한 주차면 목록에 추가
            if spot in self.ev_chargers:
                if spot not in self.available_chargers:
                    self.available_chargers.append(spot)
                if spot not in self.available_chargers_by_floor[spot[0]]:
                    self.available_chargers_by_floor[spot[0]].append(spot)
            else:
                if spot not in self.available_spots:
                    self.available_spots.append(spot)
                if spot not in self.available_spots_by_floor[spot[0]]:
                    self.available_spots_by_floor[spot[0]].append(spot)
            
            # ParkingSystem에서 주차면 해제
            floor, row, col = spot
            self.parking_system.release_parking_spot(floor, row, col)
            
            vehicle.update_state("outside")
            # 전기차인 경우 배터리 초기화
            if vehicle.vehicle_type == "ev":
                vehicle.battery_level = sample_battery_level()
            return True
        
        return False

    def handle_vehicle_exit(self, vehicle: Vehicle) -> None:
        """
        차량 출차를 처리하고 로그를 기록합니다.
        
        Args:
            vehicle: 출차할 차량
        """
        # 현재 주차 위치 확인
        spot = self.parked_vehicles.get(vehicle.vehicle_id)
        
        # 출차 처리
        if self.exit_vehicle(vehicle):
            # 메인 시뮬레이션 로그에 depart 이벤트 기록
            floor = self.convert_to_internal_floor_name(spot[0]) if spot else ""
            self.logger.log_event(
                time=self.env.now,
                vehicle_id=vehicle.vehicle_id,
                event="depart",
                floor=floor,
                pos=(spot[1], spot[2]) if spot else None,
                battery=vehicle.battery_level
            )

    def get_vehicle_position(self, vehicle_id: str) -> Optional[Tuple[str, int, int]]:
        """차량의 현재 위치 반환"""
        return self.parked_vehicles.get(vehicle_id)

    def get_parking_status(self) -> Dict[str, int]:
        """현재 주차장 상태 반환"""
        return {
            "total_parking_spots": self.total_parking_spots,
            "total_charger_spots": self.total_charger_spots,
            "total_parked": len(self.parked_vehicles),
            "double_parked": len(self.double_parked),
            "available_spots": self.get_available_spots_count(),
            "available_ev_spots": self.get_available_chargers_count()
        }

    def allocate_chargers(self, num_chargers: int) -> None:
        """
        지정된 수만큼의 주차면을 충전소로 변환합니다.
        2~3개씩 그룹화, 입구로부터 맨해튼 거리 15 이상, 각 동과의 거리의 최대-최소 차이 최소 그룹 우선.
        충전소 개수만큼 층을 랜덤하게 뽑아 분배합니다.
        Args:
            num_chargers: 할당할 충전소 수
        """
        print("\n[DEBUG] === 충전소 할당 시작 ===")
        ENTRANCE_ROW, ENTRANCE_COL = 0, 8
        DIST_THRESHOLD = 15
        # 각 동 좌표
        building_coords = self.parking_system.building_coordinates
        building_list = list(building_coords.values())
        # 기존 충전소 초기화
        self.ev_chargers.clear()
        self.available_chargers.clear()
        for floor in self.available_chargers_by_floor:
            self.available_chargers_by_floor[floor].clear()
        # 층별로 사용 가능한 주차면 수집
        floor_spots = {}
        for floor in self.available_spots_by_floor:
            spots = self.available_spots_by_floor[floor]
            floor_spots[floor] = spots
        floor_names = list(floor_spots.keys())
        # 충전소 개수만큼 층을 랜덤하게 뽑음
        random_floors = [random.choice(floor_names) for _ in range(num_chargers)]
        random.shuffle(random_floors)
        selected_spots = []
        chargers_per_floor = {f: 0 for f in floor_names}
        for f in random_floors:
            chargers_per_floor[f] += 1
        # 각 층에 대해 할당
        for floor, count in chargers_per_floor.items():
            if count == 0:
                continue
            spots = floor_spots[floor]
            # 거리 조건을 만족하는 주차면만 필터링
            valid_spots = [s for s in spots if abs(s[1] - ENTRANCE_ROW) + abs(s[2] - ENTRANCE_COL) >= DIST_THRESHOLD]
            # 연속된 주차면 그룹화
            continuous_spots = []
            current_group = []
            for i in range(len(valid_spots)):
                if not current_group:
                    current_group.append(valid_spots[i])
                else:
                    prev_spot = current_group[-1]
                    curr_spot = valid_spots[i]
                    if (prev_spot[1] == curr_spot[1] and abs(prev_spot[2] - curr_spot[2]) == 1):
                        current_group.append(curr_spot)
                    else:
                        if len(current_group) >= 2:
                            continuous_spots.append(current_group)
                        current_group = [curr_spot]
            if len(current_group) >= 2:
                continuous_spots.append(current_group)
            # 각 그룹별로 2~3개씩 뽑을 수 있는 모든 조합 생성
            group_candidates = []
            for group in continuous_spots:
                for size in [3, 2]:
                    if len(group) >= size:
                        for i in range(len(group) - size + 1):
                            candidate = group[i:i+size]
                            # 각 동과의 거리 계산
                            dists = []
                            for bx, by in building_list:
                                group_d = [abs(s[1] - bx) + abs(s[2] - by) for s in candidate]
                                dists.append(min(group_d))
                            max_min = max(dists)
                            min_min = min(dists)
                            diff = max_min - min_min
                            group_candidates.append((diff, candidate))
            # 최대-최소 차이가 가장 작은 그룹부터 우선 할당
            group_candidates.sort(key=lambda x: x[0])
            floor_selected = []
            used_spots = set()
            for diff, candidate in group_candidates:
                if len(floor_selected) + len(candidate) > count:
                    continue
                if any(tuple(s) in used_spots for s in candidate):
                    continue
                floor_selected.extend(candidate)
                used_spots.update(candidate)
                if len(floor_selected) >= count:
                    break
            # 만약 그룹에서 다 못 채우면, 남은 수만큼 랜덤하게 추가
            if len(floor_selected) < count:
                remain = count - len(floor_selected)
                single_candidates = [s for s in valid_spots if s not in floor_selected]
                if len(single_candidates) >= remain:
                    floor_selected.extend(random.sample(single_candidates, remain))
                else:
                    floor_selected.extend(single_candidates)
            selected_spots.extend(floor_selected)
        # 만약 아직도 남았다면 전체에서 거리 조건 만족하는 곳 랜덤 할당
        if len(selected_spots) < num_chargers:
            all_valid_spots = []
            for spots in floor_spots.values():
                all_valid_spots.extend([s for s in spots if abs(s[1] - ENTRANCE_ROW) + abs(s[2] - ENTRANCE_COL) >= DIST_THRESHOLD])
            remain = num_chargers - len(selected_spots)
            candidates = [s for s in all_valid_spots if s not in selected_spots]
            if len(candidates) >= remain:
                selected_spots.extend(random.sample(candidates, remain))
            else:
                selected_spots.extend(candidates)
        print(f"[DEBUG] 선택된 충전소 위치: {len(selected_spots)}개")
        # 선택된 주차면을 충전소로 변환
        floor_counts = defaultdict(int)
        for spot in selected_spots:
            floor = spot[0]
            floor_counts[floor] += 1
            if spot in self.available_spots:
                self.available_spots.remove(spot)
            if spot in self.available_spots_by_floor[floor]:
                self.available_spots_by_floor[floor].remove(spot)
            if spot in self.normal_spots:
                self.normal_spots.remove(spot)
            self.ev_chargers.add(spot)
            self.available_chargers.append(spot)
            self.available_chargers_by_floor[floor].append(spot)
            self.total_parking_spots -= 1
            self.total_charger_spots += 1
        print("\n[DEBUG] 층별 충전소 할당 결과:")
        for floor in sorted(floor_counts.keys()):
            print(f"[DEBUG] - {floor}: {floor_counts[floor]}개")
            print(f"[DEBUG]   위치: {self.available_chargers_by_floor[floor]}")
        print(f"\n[DEBUG] === 충전소 할당 완료 ({len(selected_spots)}개) ===\n")

    def convert_to_internal_floor_name(self, floor: str) -> str:
        """층 이름을 내부 형식(B1, B2, Ground 등)으로 변환"""
        return self.floor_mapping.get(floor, floor)

    def handle_vehicle_entry(self, vehicle: Vehicle) -> None:
        """
        차량 입차를 처리하고 로그를 기록합니다.
        Args:
            vehicle: 입차할 차량
        """
        # 입차 로그 기록
        self.logger.log_event(
            time=self.env.now,
            vehicle_id=vehicle.vehicle_id,
            event="arrive",
            building=vehicle.building_id
        )

        # 주차 시도
        park_success, charge_fail = self.park_vehicle(vehicle)
        if park_success:
            # 주차 성공 시 로그 기록
            spot = self.parked_vehicles[vehicle.vehicle_id]
            floor = self.convert_to_internal_floor_name(spot[0])
            self.logger.log_event(
                time=self.env.now,
                vehicle_id=vehicle.vehicle_id,
                event="park_success",
                floor=floor,
                pos=(spot[1], spot[2]),
                battery=vehicle.battery_level,
                parking_duration=vehicle.parking_duration
            )
            # EV이고 충전 필요+충전소에 주차된 경우만 charge_start
            if vehicle.vehicle_type == "ev" and vehicle.needs_charging() and spot in self.ev_chargers:
                self.logger.log_event(
                    time=self.env.now,
                    vehicle_id=vehicle.vehicle_id,
                    event="charge_start",
                    floor=floor,
                    pos=(spot[1], spot[2]),
                    battery=vehicle.battery_level
                )
        else:
            # park_fail(이중주차) 로그 기록
            self.logger.log_event(
                time=self.env.now,
                vehicle_id=vehicle.vehicle_id,
                event="park_fail"
            ) 