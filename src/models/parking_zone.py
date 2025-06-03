from dataclasses import dataclass
from typing import Dict, Tuple

@dataclass
class ParkingZone:
    """주차 구역을 나타내는 클래스"""
    is_available: bool  # 현재 사용 가능 여부
    is_ev_charger: bool  # 전기차 충전소 여부
    position: Tuple[int, int]  # 2D 좌표 (row, col)
    floor: str  # 층 정보 ("지상", "지하1", "지하2", "지하3")
    building_distances: Dict[str, float]  # 각 동으로부터의 최단 거리

    def __post_init__(self):
        """데이터 유효성 검증"""
        if not isinstance(self.position, tuple) or len(self.position) != 2:
            raise ValueError("position must be a tuple of (row, col)")
        
        if not isinstance(self.building_distances, dict):
            raise ValueError("building_distances must be a dictionary")
        
        if self.floor not in ["지상", "지하1", "지하2", "지하3"]:
            raise ValueError("Invalid floor value")

    def get_distance_to_building(self, building_id: str) -> float:
        """특정 동까지의 거리를 반환"""
        return self.building_distances.get(building_id, float('inf'))

    def get_floor_priority(self) -> int:
        """층별 우선순위 반환 (낮을수록 높은 우선순위)"""
        floor_priorities = {
            "지상": 0,
            "지하1": 1,
            "지하2": 2,
            "지하3": 3
        }
        return floor_priorities[self.floor] 