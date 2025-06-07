import json
import random
from typing import Dict, List, Optional, Tuple

class ParkingSystem:
    def __init__(self):
        self.building_coordinates = {
            "1동": (4, 2),
            "2동": (4, 8),
            "3동": (4, 14),
            "4동": (11, 2),
            "5동": (11, 8),
            "6동": (11, 14),
            "7동": (17, 3),
            "8동": (17, 14)
        }
        self.floors = ["B3", "B2", "B1", "Ground"]
        self.parking_layout = {}
        self.occupied_spots = {}  # 층별 점유된 주차면 관리
        self.load_layouts()
        self.initialize_occupied_spots()
        
        # 층 이름 매핑 추가
        self.floor_mapping = {
            'GF': 'Ground',
            'B1F': 'B1',
            'B2F': 'B2',
            'B3F': 'B3'
        }

    def convert_floor_name(self, floor: str) -> str:
        """층 이름을 내부 형식으로 변환합니다."""
        return self.floor_mapping.get(floor, floor)

    def load_layouts(self):
        """주차장 레이아웃 파일들을 로드합니다."""
        floor_files = {
            "Ground": "json/지상층_최종.json",
            "B1": "json/지하1층_최종.json",
            "B2": "json/지하2층_최종.json",
            "B3": "json/지하3층_최종.json"
        }
        
        for floor, file_path in floor_files.items():
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.parking_layout[floor] = data[floor if floor != "Ground" else "Ground"]

    def initialize_occupied_spots(self):
        """각 층별 점유 상태를 초기화합니다."""
        for floor in self.floors:
            self.occupied_spots[floor] = set()

    def calculate_distance(self, building_x: int, building_y: int, spot_x: int, spot_y: int) -> int:
        """두 지점 간의 맨해튼 거리를 계산합니다."""
        return abs(building_x - spot_x) + abs(building_y - spot_y)

    def is_valid_parking_spot(self, floor: str, x: int, y: int) -> bool:
        """해당 위치가 유효한 주차면인지 확인합니다."""
        floor = self.convert_floor_name(floor)
        if (x, y) in self.occupied_spots[floor]:
            return False
        return self.parking_layout[floor][x][y] == "주차면"

    def find_nearest_parking_spot(self, building: str) -> Optional[Dict]:
        """가장 가까운 주차면을 찾습니다."""
        building_x, building_y = self.building_coordinates[building]
        available_spots = []

        for floor in self.floors:
            layout = self.parking_layout[floor]
            for x in range(len(layout)):
                for y in range(len(layout[0])):
                    if self.is_valid_parking_spot(floor, x, y):
                        distance = self.calculate_distance(building_x, building_y, x, y)
                        available_spots.append({
                            "floor": floor,
                            "x": x,
                            "y": y,
                            "distance": distance
                        })

        if not available_spots:
            return None

        # 거리순으로 정렬
        available_spots.sort(key=lambda x: x["distance"])
        min_distance = available_spots[0]["distance"]
        
        # 최단 거리와 동일한 거리를 가진 주차면들 필터링
        closest_spots = [spot for spot in available_spots if spot["distance"] == min_distance]
        
        # 동일 거리 주차면이 여러 개인 경우 랜덤 선택
        selected_spot = random.choice(closest_spots)
        
        # 선택된 주차면을 점유 상태로 변경
        self.occupied_spots[selected_spot["floor"]].add((selected_spot["x"], selected_spot["y"]))
        
        return selected_spot

    def release_parking_spot(self, floor: str, x: int, y: int) -> None:
        """주차면을 해제합니다."""
        floor = self.convert_floor_name(floor)
        if (x, y) in self.occupied_spots[floor]:
            self.occupied_spots[floor].remove((x, y))

    def assign_parking_spot(self, vehicle: Dict) -> Optional[Dict]:
        """차량에 주차면을 할당합니다."""
        building = vehicle["building"]
        spot = self.find_nearest_parking_spot(building)
        
        if spot:
            return {
                "vehicle_id": vehicle["id"],
                "building": building,
                "assigned_spot": spot
            }
        return None

    def get_parking_status(self) -> Dict:
        """현재 주차장의 점유 상태를 반환합니다."""
        status = {}
        for floor in self.floors:
            status[floor] = {
                "total_spots": sum(row.count("주차면") for row in self.parking_layout[floor]),
                "occupied_spots": len(self.occupied_spots[floor])
            }
        return status 