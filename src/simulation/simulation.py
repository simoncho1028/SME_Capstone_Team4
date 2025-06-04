"""
주차장 시뮬레이션을 실행하는 모듈
"""
import simpy
import random
from typing import Dict, List, Optional
from pathlib import Path
import json

from src.models.vehicle import Vehicle
from src.models.vehicle_manager import VehicleManager
from src.utils.logger import SimulationLogger
from src.utils.helpers import sample_interarrival_time
from src.models.parking_manager import ParkingManager

class ParkingSimulation:
    """
    주차장 시뮬레이션을 실행하는 클래스
    """
    
    def __init__(self,
                 normal_count: int = 830,
                 ev_count: int = 36,
                 building_count: int = 8,
                 parking_capacity: int = 686,
                 charger_count: int = 10,
                 simulation_time: float = 24*60*60,  # 24시간
                 data_path: str = "data"):
        """
        시뮬레이션 객체를 초기화합니다.
        
        Args:
            normal_count: 일반 차량 수
            ev_count: EV 차량 수
            building_count: 건물 동 수
            parking_capacity: 총 주차면 수
            charger_count: EV 충전소 수
            simulation_time: 시뮬레이션 시간 (초)
            data_path: 데이터 저장 경로
        """
        # 시뮬레이션 환경 초기화
        self.env = simpy.Environment()
        
        # 리소스 초기화
        self.parking_res = simpy.Resource(self.env, capacity=parking_capacity)
        self.charger_res = simpy.Resource(self.env, capacity=charger_count)
        
        # 시뮬레이션 설정
        self.simulation_time = simulation_time
        self.data_path = Path(data_path)
        self.data_path.mkdir(parents=True, exist_ok=True)
        
        # 로거 초기화
        self.logger = SimulationLogger(
            log_file=self.data_path / "simulation_log.csv",
            stats_file=self.data_path / "simulation_stats.json"
        )
        
        # 주차장 관리자 초기화
        self.parking_manager = ParkingManager()
        self.parking_manager.set_env(self.env)
        self.parking_manager.set_logger(self.logger)
        
        # 차량 관리자 초기화
        self.vehicle_manager = VehicleManager(
            normal_count=normal_count,
            ev_count=ev_count,
            building_count=building_count,
            base_path=data_path
        )
        
        # 시뮬레이션 상태 초기화
        self.active_vehicles: Dict[str, Vehicle] = {}
        self.vehicle_processes: List[simpy.Process] = []
    
    def vehicle_arrival(self):
        """
        차량 도착 프로세스
        """
        # 모든 차량 ID 목록 가져오기
        all_vehicle_ids = list(self.vehicle_manager.vehicles.keys())
        random.shuffle(all_vehicle_ids)  # 랜덤하게 섞기
        
        for vehicle_id in all_vehicle_ids:
            # 차량 정보 가져오기
            vehicle_info = self.vehicle_manager.get_vehicle_info(vehicle_id)
            
            # 차량이 이미 활성화되어 있지 않은 경우에만 생성
            if vehicle_id not in self.active_vehicles:
                # 새 차량 생성
                vehicle = Vehicle(
                    vehicle_id=vehicle_info['vehicle_id'],
                    vehicle_type=vehicle_info['type'],
                    arrival_time=vehicle_info['arrival_time'],
                    building_id=vehicle_info['building_id'],
                    env=self.env,
                    parking_manager=self.parking_manager
                )
                
                # 차량 프로세스 시작
                self.active_vehicles[vehicle_id] = vehicle
                process = self.env.process(vehicle.process())
                self.vehicle_processes.append(process)
            
            # 다음 차량 도착까지 대기
            yield self.env.timeout(sample_interarrival_time())
    
    def run(self):
        """
        시뮬레이션을 실행합니다.
        """
        # 차량 도착 프로세스 시작
        arrival_process = self.env.process(self.vehicle_arrival())
        
        # 시뮬레이션 실행
        self.env.run(until=self.simulation_time)
        
        # 시뮬레이션 결과 저장
        self.logger.save_stats()
        
        # 차량 통계 저장
        vehicle_stats = self.vehicle_manager.get_statistics()
        stats_file = self.data_path / "vehicle_stats.json"
        with open(stats_file, "w", encoding="utf-8") as f:
            json.dump(vehicle_stats, f, ensure_ascii=False, indent=2)
        
        # 차량 정보 CSV 내보내기
        self.vehicle_manager.export_to_csv(
            self.data_path / "vehicles.csv"
        ) 