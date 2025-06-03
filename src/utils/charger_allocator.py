"""
주차장 내 충전소 위치를 관리하는 모듈
"""
import random
from typing import Dict, List, Tuple, Set
from src.config import get_all_floor_maps, CELL_PARK

class ChargerAllocator:
    """충전소 위치를 관리하는 클래스"""
    
    def __init__(self, num_chargers: int):
        """
        Args:
            num_chargers: 설치할 충전소 수
        """
        self.num_chargers = num_chargers
        self.floor_maps = get_all_floor_maps()
        self.charger_locations: Dict[str, Set[Tuple[int, int]]] = {
            floor: set() for floor in self.floor_maps.keys()
        }
        self._allocate_chargers()
    
    def _get_available_spots(self) -> List[Tuple[str, Tuple[int, int]]]:
        """모든 층의 사용 가능한 주차면 위치를 반환"""
        available_spots = []
        for floor, layout in self.floor_maps.items():
            for i, row in enumerate(layout):
                for j, cell in enumerate(row):
                    if cell == CELL_PARK:  # 주차면인 경우
                        available_spots.append((floor, (i, j)))
        return available_spots
    
    def _allocate_chargers(self) -> None:
        """충전소 위치를 무작위로 할당"""
        available_spots = self._get_available_spots()
        if len(available_spots) < self.num_chargers:
            raise ValueError(f"충전소 수({self.num_chargers})가 가용 주차면 수({len(available_spots)})를 초과합니다.")
        
        # 무작위로 충전소 위치 선택
        selected_spots = random.sample(available_spots, self.num_chargers)
        
        # 층별로 충전소 위치 저장
        for floor, pos in selected_spots:
            self.charger_locations[floor].add(pos)
    
    def is_charger(self, floor: str, pos: Tuple[int, int]) -> bool:
        """해당 위치가 충전소인지 확인"""
        return pos in self.charger_locations.get(floor, set())
    
    def get_charger_locations(self) -> Dict[str, Set[Tuple[int, int]]]:
        """모든 충전소 위치 반환"""
        return self.charger_locations.copy()
    
    def get_floor_chargers(self, floor: str) -> Set[Tuple[int, int]]:
        """특정 층의 충전소 위치 반환"""
        return self.charger_locations.get(floor, set()).copy() 