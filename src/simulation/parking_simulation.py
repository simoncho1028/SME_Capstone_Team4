"""
주차장 시뮬레이션을 실행하는 모듈
"""
import simpy
from typing import Dict, List, Optional
from datetime import datetime

from src.models.vehicle import Vehicle
from src.models.parking_manager import ParkingManager
from src.utils.logger import SimulationLogger

class ParkingSimulation:
    """주차장 시뮬레이션을 실행하는 클래스"""
    
    def __init__(self,
                 env: simpy.Environment,
                 parking_manager: ParkingManager,
                 logger: SimulationLogger):
        """
        시뮬레이션 객체를 초기화합니다.
        
        Args:
            env: SimPy 환경
            parking_manager: 주차장 관리자
            logger: 이벤트 로거
        """
        self.env = env
        self.parking_manager = parking_manager
        self.logger = logger
        
        # 현재 시뮬레이션 중인 차량들
        self.active_vehicles: Dict[str, Vehicle] = {}
        
        # 시뮬레이션 통계
        self.stats = {
            "total_entries": 0,
            "successful_parks": 0,
            "failed_parks": 0,
            "total_charges": 0
        }

    def log_event(self, vehicle: Vehicle, event: str) -> None:
        """이벤트 로깅"""
        pos = None
        if event in ["park_success", "park_fail", "charge_start", "charge_complete"]:
            pos = self.parking_manager.get_vehicle_position(vehicle.vehicle_id)
            
        self.logger.add_event(
            time=self.env.now,
            vehicle_id=vehicle.vehicle_id,
            vehicle_type=vehicle.vehicle_type,
            event=event,
            pos=pos,
            battery=vehicle.battery_level if vehicle.vehicle_type == "ev" else None
        )

    def handle_vehicle_entry(self, vehicle: Vehicle) -> None:
        """
        차량 입차 처리
        
        Args:
            vehicle: 입차하는 차량
        """
        self.stats["total_entries"] += 1
        self.log_event(vehicle, "arrive")
        
        # 즉시 주차 시도
        if self.parking_manager.park_vehicle(vehicle):
            self.stats["successful_parks"] += 1
            self.log_event(vehicle, "park_success")
            
            # 전기차 충전 시작
            if vehicle.vehicle_type == "ev" and vehicle.needs_charging():
                vehicle.start_charging()
                self.stats["total_charges"] += 1
                self.log_event(vehicle, "charge_start")
        else:
            self.stats["failed_parks"] += 1
            self.log_event(vehicle, "park_fail")

    def handle_vehicle_exit(self, vehicle: Vehicle) -> None:
        """
        차량 출차 처리
        
        Args:
            vehicle: 출차하는 차량
        """
        if self.parking_manager.exit_vehicle(vehicle):
            self.log_event(vehicle, "depart")

    def update_charging_vehicles(self) -> None:
        """충전 중인 차량들의 배터리 상태 업데이트"""
        for vehicle_id, spot in self.parking_manager.parked_vehicles.items():
            if spot in self.parking_manager.ev_chargers:
                vehicle = self.active_vehicles.get(vehicle_id)
                if vehicle and vehicle.state == "charging":
                    # 1분 단위로 배터리 업데이트
                    vehicle.update_battery(60)
                    if vehicle.battery_level >= 100.0:
                        self.log_event(vehicle, "charge_complete")

    def run(self, until: float) -> None:
        """
        시뮬레이션 실행
        
        Args:
            until: 시뮬레이션 종료 시각 (초)
        """
        def charging_update_process(env: simpy.Environment):
            """충전 상태 업데이트 프로세스"""
            while True:
                self.update_charging_vehicles()
                yield env.timeout(60)  # 1분마다 업데이트
        
        # 충전 상태 업데이트 프로세스 시작
        self.env.process(charging_update_process(self.env))
        
        # 시뮬레이션 실행
        self.env.run(until=until)

    def get_statistics(self) -> Dict:
        """시뮬레이션 통계 반환"""
        stats = self.stats.copy()
        stats.update(self.parking_manager.get_parking_status())
        return stats

    def print_summary(self) -> None:
        """시뮬레이션 결과 요약을 출력"""
        stats = self.get_statistics()
        
        print("\n=== 주차장 상태 ===")
        print(f"총 주차된 차량: {stats['total_parked']}대")
        print(f"이중주차 차량: {stats['double_parked']}대")
        print(f"사용 가능한 주차면: {stats['available_spots']}면")
        print(f"사용 가능한 충전소: {stats['available_ev_spots']}개")
        
        print("\n=== 통계 ===")
        print(f"총 입차 시도: {stats['total_entries']}회")
        print(f"성공한 주차: {stats['successful_parks']}회")
        print(f"실패한 주차: {stats['failed_parks']}회")
        print(f"총 충전 시도: {stats['total_charges']}회")

    def generate_plots(self) -> None:
        """시뮬레이션 결과를 시각화하는 그래프를 생성"""
        self.logger.generate_plots() 