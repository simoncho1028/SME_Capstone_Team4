"""
주차장 시뮬레이션 클래스
"""
import simpy
from typing import Dict, Optional
from src.models.vehicle import Vehicle
from src.models.vehicle_manager import VehicleManager
from src.utils.logger import SimulationLogger

class ParkingSimulation:
    """주차장 시뮬레이션을 실행하는 클래스"""
    
    def __init__(self,
                 env: simpy.Environment,
                 vehicle_manager: VehicleManager,
                 max_parking_spots: int = 800,
                 max_charging_spots: int = 36):
        """
        Args:
            env: SimPy 환경
            vehicle_manager: 차량 관리자
            max_parking_spots: 최대 일반 주차면 수
            max_charging_spots: 최대 충전소 수
        """
        self.env = env
        self.vehicle_manager = vehicle_manager
        self.max_parking_spots = max_parking_spots
        self.max_charging_spots = max_charging_spots
        
        # 리소스 초기화
        self.parking_spots = simpy.Resource(env, capacity=max_parking_spots)
        self.charging_spots = simpy.Resource(env, capacity=max_charging_spots)
        
        # 로거 초기화
        self.logger = SimulationLogger()
        
        # 현재 일정
        self.current_schedule: Dict[float, Vehicle] = {}
    
    def run(self, simulation_time: float = 24*60*60):
        """
        시뮬레이션 실행
        
        Args:
            simulation_time: 시뮬레이션 시간 (초)
        """
        # 일정 생성 프로세스 시작
        self.env.process(self.schedule_generator())
        # 입차 처리 프로세스 시작
        self.env.process(self.entry_processor())
        # 출차 처리 프로세스 시작
        self.env.process(self.exit_processor())
        # 시뮬레이션 실행
        self.env.run(until=simulation_time)
    
    def schedule_generator(self):
        """매일 0시에 그날의 입차 스케줄을 생성"""
        while True:
            # 현재 시각이 속한 날의 다음 날 0시를 계산
            current_time = self.env.now
            next_day_start = ((current_time // 86400) + 1) * 86400
            
            # 다음 날 0시까지 대기
            yield self.env.timeout(next_day_start - current_time)
            
            # 다음 날의 스케줄 생성
            self.current_schedule = self.vehicle_manager.generate_daily_schedule(next_day_start)
            
            # 로그 기록
            self.logger.log_event(
                "schedule_generated",
                {"time": next_day_start, "entries": len(self.current_schedule)}
            )
    
    def entry_processor(self):
        """입차 예정 차량들을 처리"""
        while True:
            current_time = self.env.now
            
            # 현재 시각에 입차할 차량이 있는지 확인
            if current_time in self.current_schedule:
                vehicle = self.current_schedule[current_time]
                
                # 주차 공간 확인
                if vehicle.type == "ev" and vehicle.battery_level < 100:
                    # 전기차이고 충전이 필요한 경우
                    if len(self.vehicle_manager.parked_vehicles) < self.max_charging_spots:
                        self.vehicle_manager.park_vehicle(vehicle, has_space=True)
                    else:
                        self.vehicle_manager.park_vehicle(vehicle, has_space=False)
                else:
                    # 일반 차량이거나 충전이 필요없는 전기차
                    if len(self.vehicle_manager.parked_vehicles) < self.max_parking_spots:
                        self.vehicle_manager.park_vehicle(vehicle, has_space=True)
                    else:
                        self.vehicle_manager.park_vehicle(vehicle, has_space=False)
                
                # 로그 기록
                self.logger.log_event(
                    "vehicle_entered",
                    {"time": current_time, "vehicle_id": vehicle.id, "state": vehicle.state}
                )
                
                # 스케줄에서 제거
                del self.current_schedule[current_time]
            
            yield self.env.timeout(1)  # 1초 대기
    
    def exit_processor(self):
        """주차된 차량들의 출차 처리"""
        while True:
            current_time = self.env.now
            
            # 주차된 차량들 확인
            for vehicle in list(self.vehicle_manager.parked_vehicles):
                entry_time = vehicle.next_entry_time - vehicle.parking_duration
                if current_time >= entry_time + vehicle.parking_duration:
                    self.vehicle_manager.exit_vehicle(vehicle)
                    # 로그 기록
                    self.logger.log_event(
                        "vehicle_exited",
                        {"time": current_time, "vehicle_id": vehicle.id}
                    )
            
            # 이중주차 차량들 확인
            for vehicle in list(self.vehicle_manager.double_parked_vehicles):
                entry_time = vehicle.next_entry_time - vehicle.parking_duration
                if current_time >= entry_time + vehicle.parking_duration:
                    self.vehicle_manager.exit_vehicle(vehicle)
                    # 로그 기록
                    self.logger.log_event(
                        "vehicle_exited",
                        {"time": current_time, "vehicle_id": vehicle.id}
                    )
            
            yield self.env.timeout(1)  # 1초 대기 