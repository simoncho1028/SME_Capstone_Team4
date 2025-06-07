import json
from typing import Dict, List, Optional
from pathlib import Path

class ParkingMapLoader:
    # 한글-영문 변환 매핑
    KOREAN_TO_ENGLISH = {
        "길": "R",      # Road
        "주차면": "P",  # Parking
        "입구": "E",    # Entrance
        "출구": "X",    # eXit
        "1동": "1동",   # 한글 그대로 유지
        "2동": "2동",   # 한글 그대로 유지
        "3동": "3동",   # 한글 그대로 유지
        "4동": "4동",   # 한글 그대로 유지
        "5동": "5동",   # 한글 그대로 유지
        "6동": "6동",   # 한글 그대로 유지
        "7동": "7동",   # 한글 그대로 유지
        "8동": "8동"    # 한글 그대로 유지
    }

    # 층별 매핑
    FLOOR_MAPPING = {
        "Ground": "GF",  # Ground Floor
        "B1": "B1F",    # Basement 1
        "B2": "B2F",    # Basement 2
        "B3": "B3F"     # Basement 3
    }

    def __init__(self, json_dir: str = "json"):
        """
        주차장 맵 로더 초기화
        
        Args:
            json_dir: JSON 파일이 있는 디렉토리 경로
        """
        self.json_dir = Path(json_dir)
        self.maps: Dict[str, List[List[str]]] = {}

    def load_all_maps(self) -> Dict[str, List[List[str]]]:
        """
        모든 층의 주차장 맵을 로드하고 변환
        
        Returns:
            Dict[str, List[List[str]]]: 층별 변환된 주차장 맵
        """
        try:
            # 각 층별 파일 로드
            files = {
                "GF": "지상층_최종.json",
                "B1F": "지하1층_최종.json",
                "B2F": "지하2층_최종.json",
                "B3F": "지하3층_최종.json"
            }

            for floor_key, filename in files.items():
                file_path = self.json_dir / filename
                if not file_path.exists():
                    print(f"[ERROR] 파일을 찾을 수 없습니다: {filename}")
                    raise FileNotFoundError(f"파일을 찾을 수 없습니다: {filename}")

                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    print(f"[DEBUG] Loaded {filename}, keys: {list(data.keys())}")
                    
                # JSON 키에 따라 데이터 추출
                map_data = None
                for key in ["Ground", "B1", "B2", "B3"]:  # 직접 키 지정
                    if key in data:
                        map_data = data[key]
                        print(f"[DEBUG] Found map data for key: {key}")
                        break
                
                if map_data is None:
                    print(f"[ERROR] 유효한 층 데이터를 찾을 수 없습니다: {filename}")
                    raise ValueError(f"유효한 층 데이터를 찾을 수 없습니다: {filename}")

                # 한글을 영문으로 변환
                converted_map = self._convert_map(map_data)
                #print(f"[DEBUG] Converted map for {floor_key}: {len(converted_map)}x{len(converted_map[0])} grid")
                self.maps[floor_key] = converted_map

            return self.maps

        except Exception as e:
            print(f"[ERROR] 맵 로딩 중 오류 발생: {str(e)}")
            raise Exception(f"맵 로딩 중 오류 발생: {str(e)}")

    def _convert_map(self, map_data: List[List[str]]) -> List[List[str]]:
        """
        한글로 된 맵 데이터를 영문 약자로 변환
        
        Args:
            map_data: 변환할 맵 데이터
            
        Returns:
            List[List[str]]: 변환된 맵 데이터
        """
        converted = []
        for i, row in enumerate(map_data):
            converted_row = []
            for j, cell in enumerate(row):
                # 매핑된 값이 있으면 사용, 없으면 원래 값 유지
                converted_cell = self.KOREAN_TO_ENGLISH.get(cell, cell)
                #print(f"[DEBUG] Converting cell at ({i}, {j}): {cell} -> {converted_cell}")  # 디버그 출력 추가
                converted_row.append(converted_cell)
            converted.append(converted_row)
        return converted

    def get_map(self, floor: str) -> Optional[List[List[str]]]:
        """
        특정 층의 맵 데이터 반환
        
        Args:
            floor: 층 식별자 (GF, B1F, B2F, B3F)
            
        Returns:
            Optional[List[List[str]]]: 해당 층의 맵 데이터
        """
        return self.maps.get(floor)

    def get_all_maps(self) -> Dict[str, List[List[str]]]:
        """
        모든 층의 맵 데이터 반환
        
        Returns:
            Dict[str, List[List[str]]]: 모든 층의 맵 데이터
        """
        return self.maps 