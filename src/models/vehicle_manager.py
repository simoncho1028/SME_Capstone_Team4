"""
차량 정보 관리 및 EV 전환 로직을 담당하는 모듈
"""
import json
import random
import numpy as np
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import pandas as pd
from src.models.vehicle import Vehicle
from src.utils.helpers import normalized_entry_ratios

class VehicleManager:
    """
    차량 정보를 관리하고 EV 전환을 처리하는 클래스
    """
    
    def __init__(self, 
                 normal_count: int = 830,
                 ev_count: int = 36,
                 building_count: int = 8,
                 base_path: str = "data"):
        """
        VehicleManager 초기화
        
        Args:
            normal_count: 일반 차량 수
            ev_count: EV 차량 수
            building_count: 건물 동 수
            base_path: 데이터 파일 저장 경로
        """
        self.normal_count = normal_count
        self.ev_count = ev_count
        self.building_count = building_count
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        
        # 차량 정보 파일 경로
        self.vehicles_file = self.base_path / "vehicles.json"
        self.converted_ev_file = self.base_path / "ev_converted_cars.json"
        
        # 차량 정보 딕셔너리
        self.vehicles: Dict[str, Dict] = {}
        
        # 전체 차량 생성
        self.outside_vehicles: List[Vehicle] = []
        self.parked_vehicles: List[Vehicle] = []
        self.double_parked_vehicles: List[Vehicle] = []
        
        # 일반 차량 생성
        for i in range(1, normal_count + 1):
            self.outside_vehicles.append(
                Vehicle(id=f"N{i}", type="normal")
            )
        
        # 전기차 생성
        for i in range(1, ev_count + 1):
            self.outside_vehicles.append(
                Vehicle(id=f"EV{i}", type="ev")
            )
        
        # 차량 순서 랜덤화
        random.shuffle(self.outside_vehicles)
        
        # 초기 차량 정보 로드 또는 생성
        if self.vehicles_file.exists():
            self._load_vehicles()
        else:
            self._initialize_vehicles()
            self._save_vehicles()
    
    def _initialize_vehicles(self) -> None:
        """
        초기 차량 정보를 생성하고 동에 랜덤하게 할당
        """
        # 모든 차량 ID 생성
        normal_ids = [f"N{i+1}" for i in range(self.normal_count)]
        ev_ids = [f"EV{i+1}" for i in range(self.ev_count)]
        all_ids = normal_ids + ev_ids
        
        # 각 동별로 할당될 차량 수 계산
        cars_per_building = len(all_ids) // self.building_count
        remaining_cars = len(all_ids) % self.building_count
        
        # 차량 ID 랜덤하게 섞기
        random.shuffle(all_ids)
        
        # 각 동에 차량 할당
        start_idx = 0
        for building in range(1, self.building_count + 1):
            # 이 동에 할당될 차량 수 계산
            cars_for_this_building = cars_per_building
            if remaining_cars > 0:
                cars_for_this_building += 1
                remaining_cars -= 1
            
            # 이 동에 차량 할당
            building_cars = all_ids[start_idx:start_idx + cars_for_this_building]
            for car_id in building_cars:
                self.vehicles[car_id] = {
                    "id": car_id,
                    "type": "EV" if car_id.startswith("EV") else "Normal",
                    "building": f"{building}동"
                }
            start_idx += cars_for_this_building
    
    def _save_vehicles(self) -> None:
        """
        차량 정보를 JSON 파일로 저장
        """
        with open(self.vehicles_file, 'w', encoding='utf-8') as f:
            json.dump({"vehicles": self.vehicles}, f, ensure_ascii=False, indent=2)
    
    def _load_vehicles(self) -> None:
        """
        저장된 차량 정보를 JSON 파일에서 로드
        """
        with open(self.vehicles_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            self.vehicles = data["vehicles"]
    
    def convert_to_ev(self, 
                      conversion_type: str = "random",
                      count: int = 0,
                      id_range: Tuple[int, int] = None) -> List[str]:
        """
        일반 차량을 EV로 전환
        
        Args:
            conversion_type: 전환 방식 ("random" 또는 "range")
            count: 랜덤 전환 시 전환할 차량 수
            id_range: ID 범위 전환 시 시작과 끝 번호
            
        Returns:
            전환된 차량 ID 목록
        """
        converted_cars = []
        normal_cars = [car_id for car_id, info in self.vehicles.items() 
                      if info["type"] == "Normal"]
        
        if conversion_type == "random" and count > 0:
            # 랜덤하게 지정된 수만큼 전환
            to_convert = random.sample(normal_cars, min(count, len(normal_cars)))
            converted_cars.extend(to_convert)
            
        elif conversion_type == "range" and id_range:
            # 지정된 범위의 차량을 전환
            start, end = id_range
            pattern = lambda x: x.startswith('N') and start <= int(x[1:]) <= end
            to_convert = [car_id for car_id in normal_cars if pattern(car_id)]
            converted_cars.extend(to_convert)
        
        # 선택된 차량들을 EV로 전환
        for car_id in converted_cars:
            self.vehicles[car_id]["type"] = "EV"
        
        # 전환 정보 저장
        if converted_cars:
            self._save_conversion_history(converted_cars)
        
        return converted_cars
    
    def _save_conversion_history(self, converted_cars: List[str]) -> None:
        """
        EV 전환 이력을 저장
        
        Args:
            converted_cars: 전환된 차량 ID 목록
        """
        # 기존 전환 이력 로드
        history = []
        if self.converted_ev_file.exists():
            with open(self.converted_ev_file, 'r', encoding='utf-8') as f:
                history = json.load(f)
        
        # 새로운 전환 정보 추가
        conversion_info = {
            "converted_at": pd.Timestamp.now().isoformat(),
            "cars": [
                {
                    "id": car_id,
                    "building": self.vehicles[car_id]["building"]
                }
                for car_id in converted_cars
            ]
        }
        history.append(conversion_info)
        
        # 전환 이력 저장
        with open(self.converted_ev_file, 'w', encoding='utf-8') as f:
            json.dump(history, f, ensure_ascii=False, indent=2)
    
    def get_vehicle_info(self, vehicle_id: str) -> Optional[Dict]:
        """
        특정 차량의 정보를 반환
        
        Args:
            vehicle_id: 차량 ID
            
        Returns:
            차량 정보 딕셔너리 또는 None
        """
        return self.vehicles.get(vehicle_id)
    
    def get_building_vehicles(self, building: str) -> List[Dict]:
        """
        특정 동의 모든 차량 정보를 반환
        
        Args:
            building: 동 이름 (예: "1동")
            
        Returns:
            차량 정보 딕셔너리 목록
        """
        return [info for info in self.vehicles.values() 
                if info["building"] == building]
    
    def get_statistics(self) -> Dict:
        """
        차량 통계 정보를 반환
        
        Returns:
            통계 정보 딕셔너리
        """
        stats = {
            "total_vehicles": len(self.vehicles),
            "normal_count": sum(1 for info in self.vehicles.values() 
                              if info["type"] == "Normal"),
            "ev_count": sum(1 for info in self.vehicles.values() 
                          if info["type"] == "EV"),
            "building_stats": {}
        }
        
        # 동별 통계
        for building in range(1, self.building_count + 1):
            building_name = f"{building}동"
            building_vehicles = self.get_building_vehicles(building_name)
            stats["building_stats"][building_name] = {
                "total": len(building_vehicles),
                "normal": sum(1 for v in building_vehicles if v["type"] == "Normal"),
                "ev": sum(1 for v in building_vehicles if v["type"] == "EV")
            }
        
        return stats
    
    def export_to_csv(self, filepath: str) -> None:
        """
        차량 정보를 CSV 파일로 내보내기
        
        Args:
            filepath: 저장할 CSV 파일 경로
        """
        df = pd.DataFrame.from_dict(self.vehicles, orient='index')
        df.to_csv(filepath, index=False, encoding='utf-8-sig')

    def generate_daily_schedule(self, current_time: float) -> Dict[float, Vehicle]:
        """
        하루 동안의 입차 스케줄을 생성
        
        Args:
            current_time: 현재 시각 (초)
            
        Returns:
            Dict[float, Vehicle]: 입차 시각별 차량 매핑
        """
        # 오늘 입차할 총 차량 수 결정 (전체의 80-90%)
        total_vehicles = len(self.outside_vehicles)
        daily_ratio = random.uniform(0.8, 0.9)
        daily_entries = int(total_vehicles * daily_ratio)
        
        # 시간대별 차량 수 계산
        hourly_entries = []
        for ratio in normalized_entry_ratios:
            # 기본 차량 수 계산
            base_count = int(daily_entries * ratio)
            # ±10% 노이즈 추가
            noise = random.uniform(-0.1, 0.1)
            count = max(0, int(base_count * (1 + noise)))
            hourly_entries.append(count)
        
        # 각 시간대 내에서 균등하게 입차 시각 분배
        schedule = {}
        available_vehicles = [v for v in self.outside_vehicles if v.state == "outside"]
        current_hour = 0
        
        for hour_entries in hourly_entries:
            if not available_vehicles or hour_entries == 0:
                continue
            
            # 시간대 내에서 균등하게 분배
            hour_start = current_time + (current_hour * 3600)
            for _ in range(min(hour_entries, len(available_vehicles))):
                if not available_vehicles:
                    break
                entry_time = hour_start + random.uniform(0, 3600)
                vehicle = available_vehicles.pop()
                schedule[entry_time] = vehicle
                vehicle.next_entry_time = entry_time
                vehicle.set_next_parking_duration()
            
            current_hour += 1
        
        return schedule
    
    def park_vehicle(self, vehicle: Vehicle, has_space: bool = True) -> None:
        """
        차량을 주차 처리
        
        Args:
            vehicle: 주차할 차량
            has_space: 주차 공간 여부
        """
        if vehicle not in self.outside_vehicles:
            return
            
        self.outside_vehicles.remove(vehicle)
        if has_space:
            vehicle.state = "parked"
            self.parked_vehicles.append(vehicle)
        else:
            vehicle.state = "double_parked"
            self.double_parked_vehicles.append(vehicle)
    
    def exit_vehicle(self, vehicle: Vehicle) -> None:
        """
        차량을 출차 처리
        
        Args:
            vehicle: 출차할 차량
        """
        if vehicle in self.parked_vehicles:
            self.parked_vehicles.remove(vehicle)
        elif vehicle in self.double_parked_vehicles:
            self.double_parked_vehicles.remove(vehicle)
        else:
            return
            
        vehicle.state = "outside"
        self.outside_vehicles.append(vehicle)
    
    def get_vehicle_counts(self) -> Dict[str, int]:
        """현재 차량 상태별 수를 반환"""
        return {
            "outside": len(self.outside_vehicles),
            "parked": len(self.parked_vehicles),
            "double_parked": len(self.double_parked_vehicles)
        } 