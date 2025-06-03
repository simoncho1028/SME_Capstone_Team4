import random
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

from src.models.vehicle import Vehicle
from src.models.parking_zone import ParkingZone
from src.config import PARKING_MAPS, CELL_PARK, CELL_CHARGER

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
        for floor, floor_map in PARKING_MAPS.items():
            self.available_spots_by_floor[floor] = []
            self.available_chargers_by_floor[floor] = []
            
            for row, line in enumerate(floor_map):
                for col, cell in enumerate(line):
                    if cell in [CELL_PARK, CELL_CHARGER]:  # 모든 주차 가능한 공간을 일단 일반 주차면으로 취급
                        spot = (floor, row, col)
                        self.available_spots.append(spot)
                        self.available_spots_by_floor[floor].append(spot)
    
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
            all_spots.extend([(spot, floor) for spot in self.available_spots_by_floor[floor]])
        
        # 충전소로 변환할 주차면 선택
        selected_spots = random.sample(all_spots, min(num_chargers, len(all_spots)))
        
        # 선택된 주차면을 충전소로 변환
        for spot, floor in selected_spots:
            # 일반 주차면 목록에서 제거
            if spot in self.available_spots:
                self.available_spots.remove(spot)
            if spot in self.available_spots_by_floor[floor]:
                self.available_spots_by_floor[floor].remove(spot)
            
            # 충전소 목록에 추가
            self.ev_chargers.add(spot)
            self.available_chargers.append(spot)
            self.available_chargers_by_floor[floor].append(spot)
            
        print(f"[INFO] {len(selected_spots)}개의 충전소가 할당되었습니다.")
        for floor in self.available_chargers_by_floor:
            count = len(self.available_chargers_by_floor[floor])
            if count > 0:
                print(f"  - {floor}: {count}개")
    
    def get_available_floor(self) -> str:
        """
        가장 여유 있는 층을 반환합니다.
        
        Returns:
            str: 층 식별자 (예: 'B1F', 'B2F', 'B3F')
        """
        # 각 층별 사용 가능한 주차면 비율 계산
        floor_ratios = {}
        for floor in self.available_spots_by_floor:
            total_spots = len(self.available_spots_by_floor[floor]) + len(self.available_chargers_by_floor[floor])
            if total_spots > 0:  # 0으로 나누기 방지
                available = (len([s for s in self.available_spots_by_floor[floor] if s not in self.parking_spots]) +
                           len([s for s in self.available_chargers_by_floor[floor] if s not in self.parking_spots]))
                floor_ratios[floor] = available / total_spots
            else:
                floor_ratios[floor] = 0
        
        # 가장 여유 있는 층 반환
        return max(floor_ratios.items(), key=lambda x: x[1])[0]
    
    def get_vehicle_position(self, vehicle_id: str) -> Optional[tuple]:
        """
        차량의 현재 위치를 반환합니다.
        
        Args:
            vehicle_id: 차량 ID
            
        Returns:
            (floor, row, col) 튜플 또는 None (차량이 주차되어 있지 않은 경우)
        """
        return self.parked_vehicles.get(vehicle_id)
    
    def park_vehicle(self, vehicle: Vehicle) -> bool:
        """
        차량을 주차합니다.
        
        Args:
            vehicle: 주차할 차량
            
        Returns:
            주차 성공 여부
        """
        # 이미 주차된 차량인 경우
        if vehicle.vehicle_id in self.parked_vehicles:
            return False
        
        # 가장 여유 있는 층 선택
        target_floor = self.get_available_floor()
        
        # 전기차의 경우 충전소에 우선 배정
        if vehicle.vehicle_type == "ev" and vehicle.needs_charging():
            # 선택된 층의 사용 가능한 충전소 확인
            available_chargers = [spot for spot in self.available_chargers_by_floor[target_floor]
                                if spot not in self.parking_spots]
            
            if available_chargers:
                spot = random.choice(available_chargers)
                self.parked_vehicles[vehicle.vehicle_id] = spot
                self.parking_spots[spot] = vehicle.vehicle_id
                vehicle.update_state("parked")
                return True
            
            # 다른 층의 충전소도 확인
            for floor in self.available_chargers_by_floor:
                if floor == target_floor:
                    continue
                available_chargers = [spot for spot in self.available_chargers_by_floor[floor]
                                    if spot not in self.parking_spots]
                if available_chargers:
                    spot = random.choice(available_chargers)
                    self.parked_vehicles[vehicle.vehicle_id] = spot
                    self.parking_spots[spot] = vehicle.vehicle_id
                    vehicle.update_state("parked")
                    return True
        
        # 일반 주차면에 배정
        available_spots = [spot for spot in self.available_spots_by_floor[target_floor]
                         if spot not in self.parking_spots]
        
        if available_spots:
            spot = random.choice(available_spots)
            self.parked_vehicles[vehicle.vehicle_id] = spot
            self.parking_spots[spot] = vehicle.vehicle_id
            vehicle.update_state("parked")
            return True
        
        # 다른 층의 주차면도 확인
        for floor in self.available_spots_by_floor:
            if floor == target_floor:
                continue
            available_spots = [spot for spot in self.available_spots_by_floor[floor]
                             if spot not in self.parking_spots]
            if available_spots:
                spot = random.choice(available_spots)
                self.parked_vehicles[vehicle.vehicle_id] = spot
                self.parking_spots[spot] = vehicle.vehicle_id
                vehicle.update_state("parked")
                return True
        
        # 이중주차 시도
        if len(self.double_parked) < len(self.available_spots) * 0.2:  # 최대 20%까지 이중주차 허용
            self.double_parked.add(vehicle.vehicle_id)
            vehicle.update_state("double_parked")
            return True
        
        return False
    
    def exit_vehicle(self, vehicle: Vehicle) -> bool:
        """
        차량을 출차합니다.
        
        Args:
            vehicle: 출차할 차량
            
        Returns:
            출차 성공 여부
        """
        # 이중주차된 차량 처리
        if vehicle.vehicle_id in self.double_parked:
            self.double_parked.remove(vehicle.vehicle_id)
            vehicle.update_state("outside")
            return True
        
        # 일반 주차된 차량 처리
        spot = self.parked_vehicles.get(vehicle.vehicle_id)
        if spot:
            floor, row, col = spot
            del self.parked_vehicles[vehicle.vehicle_id]
            if spot in self.parking_spots:
                del self.parking_spots[spot]
            
            # 충전소였던 경우
            if spot in self.ev_chargers:
                self.available_chargers.append(spot)
                self.available_chargers_by_floor[floor].append(spot)
            else:
                self.available_spots.append(spot)
                self.available_spots_by_floor[floor].append(spot)
            
            vehicle.update_state("outside")
            return True
        
        return False
    
    def get_parking_status(self) -> Dict[str, int]:
        """
        현재 주차장 상태를 반환합니다.
        
        Returns:
            주차장 상태 정보를 담은 딕셔너리
        """
        return {
            "total_parked": len(self.parked_vehicles),
            "double_parked": len(self.double_parked),
            "available_spots": len(self.available_spots),
            "available_ev_spots": len(self.available_chargers)
        } 