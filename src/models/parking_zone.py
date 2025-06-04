from typing import Dict, Tuple

class ParkingZone:
    """주차구역 클래스"""
    
    FLOOR_PRIORITY = {
        "지상": 0,
        "지하1": 1,
        "지하2": 2,
        "지하3": 3
    }
    
    def __init__(self, 
                 is_available: bool,
                 is_ev_charger: bool,
                 position: Tuple[int, int],
                 floor: str,
                 building_distances: Dict[str, float]):
        """
        주차구역 초기화
        
        Args:
            is_available: 사용 가능 여부
            is_ev_charger: 전기차 충전소 여부
            position: (x, y) 좌표
            floor: 층 정보 ("지상", "지하1", "지하2", "지하3")
            building_distances: 각 건물까지의 거리 {building_id: distance}
        """
        self.is_available = is_available
        self.is_ev_charger = is_ev_charger
        self.position = position
        self.floor = floor
        self.building_distances = building_distances
    
    def get_distance_to_building(self, building_id: str) -> float:
        """특정 건물까지의 거리 반환"""
        return self.building_distances.get(building_id, float('inf'))
    
    def get_floor_priority(self) -> int:
        """층 우선순위 반환 (낮을수록 높은 우선순위)"""
        return self.FLOOR_PRIORITY.get(self.floor, float('inf'))
    
    def __str__(self) -> str:
        """문자열 표현"""
        return f"ParkingZone(floor={self.floor}, pos={self.position}, " \
               f"available={self.is_available}, ev_charger={self.is_ev_charger})" 