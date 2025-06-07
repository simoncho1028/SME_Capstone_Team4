class ParkingSystem:
    def __init__(self):
        """주차 시스템 초기화"""
        self.occupied_spots = {}  # (floor, row, col) -> vehicle_id
        self.charging_spots = {}  # (floor, row, col) -> vehicle_id
        self.charging_vehicles = set()  # 충전 중인 차량 ID 집합
        
        # 층별 주차면 관리
        self.spots_by_floor = {}  # floor -> [(row, col), ...]
        self.chargers_by_floor = {}  # floor -> [(row, col), ...]
        
        # 전체 사용 가능한 주차면과 충전소
        self.available_spots = []  # [(floor, row, col), ...]
        self.available_chargers = []  # [(floor, row, col), ...]
        
        # 층 이름 매핑 추가
        self.floor_mapping = {
            'B1F': 'B1',
            'B2F': 'B2',
            'B3F': 'B3',
            'GF': 'Ground'
        }
        
    def convert_floor_name(self, floor: str) -> str:
        """층 이름을 'B1', 'B2', 'B3' 형식으로 변환"""
        return self.floor_mapping.get(floor, floor)
        
    def initialize_parking_map(self, parking_map: dict):
        """주차장 맵 초기화"""
        for floor, floor_data in parking_map.items():
            floor = self.convert_floor_name(floor)  # 층 이름 변환
            self.spots_by_floor[floor] = []
            self.chargers_by_floor[floor] = []
            
            for row_idx, row in enumerate(floor_data):
                for col_idx, spot_type in enumerate(row):
                    if spot_type == 1:  # 일반 주차면
                        self.spots_by_floor[floor].append((row_idx, col_idx))
                        self.available_spots.append((floor, row_idx, col_idx))
                    elif spot_type == 2:  # 충전소
                        self.chargers_by_floor[floor].append((row_idx, col_idx))
                        self.available_chargers.append((floor, row_idx, col_idx))
                        
    def assign_parking_spot(self, vehicle_id: str, is_ev: bool) -> tuple:
        """주차 공간 할당"""
        if is_ev:
            if not self.available_chargers:
                return None
            floor, row, col = self.available_chargers.pop(0)
            self.charging_spots[(floor, row, col)] = vehicle_id
            return floor, row, col
        else:
            if not self.available_spots:
                return None
            floor, row, col = self.available_spots.pop(0)
            self.occupied_spots[(floor, row, col)] = vehicle_id
            return floor, row, col 