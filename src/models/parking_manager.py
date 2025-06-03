import random
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

from src.models.vehicle import Vehicle
from src.models.parking_zone import ParkingZone

class ParkingManager:
    """주차장 관리자 클래스"""
    
    def __init__(self):
        """주차장 관리자 초기화"""
        self.parked_vehicles = {}  # vehicle_id -> (floor, zone)
        self.double_parked = set()  # 이중주차된 차량 ID 집합
        self.parking_spots = {}  # (floor, zone) -> vehicle_id
        self.ev_chargers = set()  # 충전소 위치 집합
        
        # 주차장 초기화
        self.total_spots = 686
        self.total_chargers = 36
        self.available_spots = self.total_spots - self.total_chargers
        self.available_chargers = self.total_chargers
    
    def get_vehicle_position(self, vehicle_id: str) -> Optional[tuple]:
        """
        차량의 현재 위치를 반환합니다.
        
        Args:
            vehicle_id: 차량 ID
            
        Returns:
            (floor, zone) 튜플 또는 None (차량이 주차되어 있지 않은 경우)
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
        
        # 전기차의 경우 충전소에 우선 배정
        if vehicle.vehicle_type == "ev" and vehicle.needs_charging() and self.available_chargers > 0:
            # 임시로 첫 번째 층, 첫 번째 구역에 배정
            spot = (1, self.available_chargers)
            self.parked_vehicles[vehicle.vehicle_id] = spot
            self.parking_spots[spot] = vehicle.vehicle_id
            self.ev_chargers.add(spot)
            self.available_chargers -= 1
            vehicle.update_state("parked")
            return True
        
        # 일반 주차면에 배정
        if self.available_spots > 0:
            # 임시로 첫 번째 층, 첫 번째 구역에 배정
            spot = (1, self.total_spots - self.available_spots + 1)
            self.parked_vehicles[vehicle.vehicle_id] = spot
            self.parking_spots[spot] = vehicle.vehicle_id
            self.available_spots -= 1
            vehicle.update_state("parked")
            return True
        
        # 이중주차 시도
        if len(self.double_parked) < self.total_spots * 0.2:  # 최대 20%까지 이중주차 허용
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
            del self.parked_vehicles[vehicle.vehicle_id]
            if spot in self.parking_spots:
                del self.parking_spots[spot]
            
            # 충전소였던 경우
            if spot in self.ev_chargers:
                self.available_chargers += 1
            else:
                self.available_spots += 1
            
            vehicle.update_state("outside")
            return True
        
        return False
    
    def add_parking_zone(self, zone: ParkingZone) -> None:
        """주차 구역 추가"""
        if zone.floor not in self.parking_zones:
            raise ValueError(f"Invalid floor: {zone.floor}")
        self.parking_zones[zone.floor].append(zone)

    def find_nearest_parking_zone(self, vehicle: Vehicle) -> Optional[Tuple[str, ParkingZone]]:
        """
        차량에 가장 적합한 주차 구역을 찾습니다.
        
        Args:
            vehicle: 주차할 차량
            
        Returns:
            Optional[Tuple[str, ParkingZone]]: (층, 주차구역) 또는 None
        """
        candidates = []  # (거리, 층 우선순위, 주차구역, 층) 튜플 리스트
        
        for floor, zones in self.parking_zones.items():
            for zone in zones:
                # 주차 가능 여부 확인
                if not zone.is_available:
                    continue
                    
                # 전기차는 충전기가 있는 구역만 확인
                if vehicle.vehicle_type == "ev" and vehicle.needs_charging():
                    if not zone.is_ev_charger:
                        continue
                
                # 해당 건물까지의 거리 계산
                distance = zone.get_distance_to_building(vehicle.building_id)
                floor_priority = zone.get_floor_priority()
                
                candidates.append((distance, floor_priority, zone, floor))
        
        if not candidates:
            return None
            
        # 거리와 층 우선순위로 정렬
        candidates.sort(key=lambda x: (x[0], x[1]))
        
        # 최단 거리와 최우선 층을 가진 후보들 선택
        min_distance = candidates[0][0]
        min_priority = candidates[0][1]
        best_candidates = [
            (zone, floor) for d, p, zone, floor in candidates 
            if d == min_distance and p == min_priority
        ]
        
        # 동일한 조건의 후보들 중 랜덤 선택
        selected_zone, selected_floor = random.choice(best_candidates)
        return (selected_floor, selected_zone)

    def get_parking_status(self) -> Dict[str, int]:
        """
        현재 주차장 상태를 반환합니다.
        
        Returns:
            주차장 상태 정보를 담은 딕셔너리
        """
        return {
            "total_parked": len(self.parked_vehicles),
            "double_parked": len(self.double_parked),
            "available_spots": self.available_spots,
            "available_ev_spots": self.available_chargers
        } 