import random
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

from src.models.vehicle import Vehicle
from src.config import PARKING_MAPS, CELL_PARK, CELL_CHARGER
from parking_system import ParkingSystem  # ParkingSystem import 추가

class ParkingManager:
    """주차장 관리자 클래스"""
    
    def __init__(self):
        """주차장 관리자 초기화"""
        self.parked_vehicles = {}  # vehicle_id -> (floor, row, col)
        self.double_parked = set()  # 이중주차된 차량 ID 집합
        self.parking_spots = {}  # (floor, row, col) -> vehicle_id
        self.ev_chargers = set()  # 충전소 위치 집합
        
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
    
    def _initialize_parking_map(self):
        """주차장 맵 초기화"""
        for floor, floor_map in PARKING_MAPS.items():
            self.available_spots_by_floor[floor] = []
            self.available_chargers_by_floor[floor] = []
            
            for row, line in enumerate(floor_map):
                for col, cell in enumerate(line):
                    spot = (floor, row, col)
                    if cell == CELL_PARK:  # 일반 주차면
                        self.available_spots.append(spot)
                        self.available_spots_by_floor[floor].append(spot)
                        self.total_parking_spots += 1
                    elif cell == CELL_CHARGER:  # 충전소
                        self.available_chargers.append(spot)
                        self.available_chargers_by_floor[floor].append(spot)
                        self.ev_chargers.add(spot)
                        self.total_charger_spots += 1
    
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
    
    def park_vehicle(self, vehicle: Vehicle) -> bool:
        """
        차량을 주차합니다.
        
        Args:
            vehicle: 주차할 차량
            
        Returns:
            bool: 주차 성공 여부
        """
        # 이미 주차된 차량인 경우
        if vehicle.vehicle_id in self.parked_vehicles:
            return False
        
        # 전기차의 경우 충전소에 우선 배정
        if vehicle.vehicle_type == "ev" and vehicle.needs_charging():
            # 사용 가능한 충전소 찾기
            available_chargers = [spot for spot in self.available_chargers if self.is_spot_available(spot)]
            
            if available_chargers:
                spot = random.choice(available_chargers)
                self.parked_vehicles[vehicle.vehicle_id] = spot
                self.parking_spots[spot] = vehicle.vehicle_id
                vehicle.update_state("parked")
                if self.env:
                    vehicle.start_charging(self.env.now)
                return True
        
        # 일반 주차면에 최단 거리 기반으로 배정
        spot_assignment = self.parking_system.assign_parking_spot({
            "id": vehicle.vehicle_id,
            "building": vehicle.building_id
        })
        
        if spot_assignment:
            spot = spot_assignment["assigned_spot"]
            spot_tuple = (spot["floor"], spot["x"], spot["y"])
            
            # 해당 spot이 이미 사용 중인지 확인
            if self.is_spot_available(spot_tuple):
                self.parked_vehicles[vehicle.vehicle_id] = spot_tuple
                self.parking_spots[spot_tuple] = vehicle.vehicle_id
                vehicle.update_state("parked")
                return True
        
        return False

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
            # 전기차인 경우 배터리 랜덤 초기화
            if vehicle.vehicle_type == "ev":
                vehicle.battery_level = random.uniform(20.0, 80.0)
            return True
        
        # 일반 주차된 차량 처리
        spot = self.parked_vehicles.get(vehicle.vehicle_id)
        if spot:
            # 주차면에서 차량 정보 제거
            del self.parked_vehicles[vehicle.vehicle_id]
            if spot in self.parking_spots:
                del self.parking_spots[spot]
            
            vehicle.update_state("outside")
            # 전기차인 경우 배터리 랜덤 초기화
            if vehicle.vehicle_type == "ev":
                vehicle.battery_level = random.uniform(20.0, 80.0)
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
            self.logger.log_event(
                time=self.env.now,
                vehicle_id=vehicle.vehicle_id,
                event="depart",
                floor=spot[0] if spot else "",
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
        
        Args:
            num_chargers: 할당할 충전소 수
        """
        # 기존 충전소 초기화
        self.ev_chargers.clear()
        self.available_chargers.clear()
        for floor in self.available_chargers_by_floor:
            self.available_chargers_by_floor[floor].clear()
        
        # 모든 사용 가능한 주차면 중에서 랜덤하게 선택
        all_spots = []
        for floor in self.available_spots_by_floor:
            all_spots.extend(self.available_spots_by_floor[floor])
        
        # 충전소로 변환할 주차면 선택
        selected_spots = random.sample(all_spots, min(num_chargers, len(all_spots)))
        
        # 선택된 주차면을 충전소로 변환
        for spot in selected_spots:
            floor = spot[0]  # (floor, row, col)
            
            # 일반 주차면 목록에서 제거
            if spot in self.available_spots:
                self.available_spots.remove(spot)
            if spot in self.available_spots_by_floor[floor]:
                self.available_spots_by_floor[floor].remove(spot)
            
            # 충전소 목록에 추가
            self.ev_chargers.add(spot)
            self.available_chargers.append(spot)
            self.available_chargers_by_floor[floor].append(spot)
            
            # 전체 주차면/충전소 수 업데이트
            self.total_parking_spots -= 1
            self.total_charger_spots += 1
            
        print(f"[INFO] {len(selected_spots)}개의 충전소가 할당되었습니다.")
        for floor in self.available_chargers_by_floor:
            count = len(self.available_chargers_by_floor[floor])
            if count > 0:
                print(f"  - {floor}: {count}개")

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
        if self.park_vehicle(vehicle):
            # 주차 성공 시 로그 기록
            spot = self.parked_vehicles[vehicle.vehicle_id]
            
            self.logger.log_event(
                time=self.env.now,
                vehicle_id=vehicle.vehicle_id,
                event="park_success",
                floor=spot[0],
                pos=(spot[1], spot[2]),
                battery=vehicle.battery_level,
                parking_duration=vehicle.parking_duration  # 주차 예정 시간 기록
            )
        else:
            # 주차 실패 시 로그 기록
            self.logger.log_event(
                time=self.env.now,
                vehicle_id=vehicle.vehicle_id,
                event="park_fail"
            ) 