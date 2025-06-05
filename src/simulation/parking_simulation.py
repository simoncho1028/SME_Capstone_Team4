"""
주차장 시뮬레이션을 실행하는 모듈
"""
import simpy
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import os
import random
import numpy as np

from src.models.vehicle import Vehicle
from src.models.parking_manager import ParkingManager
from src.utils.logger import SimulationLogger

class ParkingSimulation:
    """주차장 시뮬레이션을 실행하는 클래스"""
    
    def __init__(self,
                 env: simpy.Environment,
                 parking_manager: ParkingManager,
                 logger: SimulationLogger,
                 total_vehicle_count: int = 866):
        """
        시뮬레이션 객체를 초기화합니다.
        
        Args:
            env: SimPy 환경
            parking_manager: 주차장 관리자
            logger: 이벤트 로깅을 위한 로거 객체
            total_vehicle_count: 전체 차량 수 (CLI에서 지정된 --normal + --ev 값)
        """
        self.env = env
        self.parking_manager = parking_manager
        self.logger = logger
        self.total_vehicle_count = total_vehicle_count
        
        # parking_manager에 env 설정
        self.parking_manager.set_env(env)
        
        # 현재 시뮬레이션 중인 차량들
        self.active_vehicles: Dict[str, Vehicle] = {}
        
        # 대기 중인 차량들 (아직 입차하지 않은 차량들)
        self.outside_vehicles: Dict[str, Vehicle] = {}
        
        # 시뮬레이션 통계
        self.stats = {
            "total_entries": 0,
            "successful_parks": 0,
            "failed_parks": 0,
            "total_charges": 0
        }
        
        # 하루 단위 입차 계획
        self.daily_entry_plan: List[Tuple[float, Vehicle]] = []

    def log_event(self, vehicle: Vehicle, event: str) -> None:
        """이벤트 로깅"""
        pos = None
        floor = None
        if event in ["park_success", "park_fail", "charge_start", "charge_complete"]:
            vehicle_pos = self.parking_manager.get_vehicle_position(vehicle.vehicle_id)
            if vehicle_pos:
                floor = vehicle_pos[0]  # 층 정보
                pos = (vehicle_pos[1], vehicle_pos[2])  # (row, col)
            
        self.logger.add_event(
            time=self.env.now,
            vehicle_id=vehicle.vehicle_id,
            vehicle_type=vehicle.vehicle_type,
            event=event,
            pos=pos,
            battery=vehicle.battery_level if vehicle.vehicle_type == "ev" else None,
            building=vehicle.building_id,
            floor=floor
        )

    def handle_vehicle_entry(self, vehicle: Vehicle) -> None:
        """
        차량 입차 처리
        
        Args:
            vehicle: 입차하는 차량
        """
        self.stats["total_entries"] += 1
        self.log_event(vehicle, "arrive")
        
        # 차량을 active_vehicles에 추가
        self.active_vehicles[vehicle.vehicle_id] = vehicle
        
        # 즉시 주차 시도
        if self.parking_manager.park_vehicle(vehicle):
            self.stats["successful_parks"] += 1
            self.log_event(vehicle, "park_success")
            
            # 주차 성공 시 출차 프로세스 시작
            self.env.process(self.vehicle_exit_process(vehicle))
            
            # 전기차이고 충전이 필요하며 충전소에 주차된 경우에만 충전 시작
            if (vehicle.vehicle_type == "ev" and 
                vehicle.needs_charging() and 
                self.parking_manager.get_vehicle_position(vehicle.vehicle_id) in self.parking_manager.ev_chargers):
                vehicle.start_charging(self.env.now)
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
            # active_vehicles에서 제거
            if vehicle.vehicle_id in self.active_vehicles:
                del self.active_vehicles[vehicle.vehicle_id]
            
            # 차량을 outside_vehicles로 돌려보내서 나중에 다시 입차 가능하게 함
            self.outside_vehicles[vehicle.vehicle_id] = vehicle

    def update_charging_vehicles(self) -> None:
        """충전 중인 차량들의 배터리 상태 업데이트"""
        for vehicle_id, spot in self.parking_manager.parked_vehicles.items():
            if spot in self.parking_manager.ev_chargers:
                vehicle = self.active_vehicles.get(vehicle_id)
                if vehicle and vehicle.state == "charging" and vehicle.battery_level < 100.0:
                    # 현재 배터리 레벨 저장
                    old_battery = vehicle.battery_level
                    
                    # 1분 단위로 배터리 업데이트
                    vehicle.update_charging(self.env.now)
                    
                    # 배터리 레벨이 변경되었으면 로그에 기록
                    if vehicle.battery_level > old_battery:
                        self.log_event(vehicle, "charge_update")
                    
                    # 충전 완료 시
                    if vehicle.battery_level >= 100.0:
                        self.log_event(vehicle, "charge_complete")
                        # 충전 완료 후 상태 변경
                        vehicle.update_state("parked")

    def plan_daily_entries(self, current_time: float) -> None:
        """
        하루 단위 입차 계획을 수립합니다.
        
        Args:
            current_time: 현재 시뮬레이션 시간
        """
        from src.config import normalized_entry_ratios
        
        # 하루의 시작 시간 계산 (현재 시간의 0시)
        day_start = current_time - (current_time % 86400)
        
        # 입차할 총 차량 수 결정 (전체 차량 수의 80~90%)
        total_entries = int(self.total_vehicle_count * np.random.uniform(0.8, 0.9))
        
        # 시간대별 입차량 계산
        hourly_entries = []
        for ratio in normalized_entry_ratios:
            # ±10% 노이즈 추가
            noise = np.random.uniform(0.9, 1.1)
            entries = int(total_entries * ratio * noise)
            hourly_entries.append(entries)
        
        # 각 시간대별로 차량 입차 시간 샘플링
        self.daily_entry_plan.clear()
        for hour, entries in enumerate(hourly_entries):
            if entries > 0:
                # 해당 시간대에 uniform하게 입차 시간 분배
                hour_start = day_start + (hour * 3600)
                entry_times = np.random.uniform(hour_start, hour_start + 3600, entries)
                
                # 입차할 차량 선택 및 계획 수립
                for entry_time in sorted(entry_times):
                    if self.outside_vehicles:  # 대기 중인 차량이 있는 경우
                        vehicle_id = random.choice(list(self.outside_vehicles.keys()))
                        vehicle = self.outside_vehicles.pop(vehicle_id)
                        self.daily_entry_plan.append((entry_time, vehicle))
    
    def daily_planning_process(self):
        """매일 0시마다 새로운 입차 계획을 수립하는 프로세스"""
        while True:
            # 현재 시간의 다음 날 0시까지 대기
            next_day = (self.env.now - (self.env.now % 86400)) + 86400
            yield self.env.timeout(next_day - self.env.now)
            
            # 새로운 하루의 입차 계획 수립
            self.plan_daily_entries(self.env.now)
    
    def vehicle_entry_process(self):
        """입차 계획에 따라 차량을 입차시키는 프로세스"""
        while True:
            if self.daily_entry_plan:
                # 가장 빠른 입차 예정 차량 확인
                next_entry_time, vehicle = self.daily_entry_plan[0]
                
                # 입차 시간까지 대기
                if next_entry_time > self.env.now:
                    yield self.env.timeout(next_entry_time - self.env.now)
                
                # 입차 처리
                self.handle_vehicle_entry(vehicle)
                
                # 처리된 차량 제거
                self.daily_entry_plan.pop(0)
            else:
                # 다음 계획이 수립될 때까지 1분 대기
                yield self.env.timeout(60)

    def vehicle_exit_process(self, vehicle: Vehicle):
        """
        차량의 주차 시간 후 자동 출차 프로세스
        
        Args:
            vehicle: 출차할 차량
        """
        # 주차 시간만큼 대기
        yield self.env.timeout(vehicle.parking_duration)
        
        # 출차 처리
        self.handle_vehicle_exit(vehicle)

    def run(self, until: float) -> None:
        """
        시뮬레이션 실행
        
        Args:
            until: 시뮬레이션 종료 시각 (초)
        """
        # 초기 입차 계획 수립
        self.plan_daily_entries(self.env.now)
        
        # 일일 계획 수립 프로세스 시작
        self.env.process(self.daily_planning_process())
        
        # 차량 입차 프로세스 시작
        self.env.process(self.vehicle_entry_process())
        
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

    def save_summary_to_file(self, output_dir: str) -> None:
        """
        시뮬레이션 결과 요약을 파일로 저장합니다.
        
        Args:
            output_dir: 저장할 디렉토리 경로
        """
        # 현재 시간으로 파일명 생성
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(output_dir, f"simulation_summary_{timestamp}.txt")
        
        stats = self.get_statistics()
        
        # 시뮬레이션 진행 시간을 일/시간/분/초로 변환
        def format_time(seconds):
            days = int(seconds // 86400)
            hours = int((seconds % 86400) // 3600)
            minutes = int((seconds % 3600) // 60)
            secs = int(seconds % 60)
            return f"{days}일 {hours}시간 {minutes}분 {secs}초"
        
        sim_time_str = format_time(self.env.now)
        
        # 출차 횟수 계산
        df = self.logger.get_dataframe()
        depart_count = len(df[df.event == "depart"]) if not df.empty else 0
        
        # 실패한 충전 시도 계산
        charge_fail_count = len(df[df.event == "charge_fail"]) if not df.empty else 0
        
        with open(filename, "w", encoding="utf-8") as f:
            f.write("==== 시뮬레이션 결과 ====\n")
            f.write(f"총 진행 시간: {sim_time_str}\n\n")
            
            f.write("==== 주차장 세팅 ====\n")
            f.write(f"전체 주차면: {stats['total_parking_spots']}면 (일반 주차면)\n")
            f.write(f"전체 충전소: {stats['total_charger_spots']}개\n\n")
            
            f.write("=== 주차장 상태 (현재) ===\n")
            f.write(f"({sim_time_str}) 동안 진행 후 현재 상황\n")
            f.write(f"총 주차된 차량: {stats['total_parked']}대\n")
            f.write(f"이중주차 차량: {stats['double_parked']}대\n")
            f.write(f"사용 가능한 주차면: {stats['available_spots']}면\n")
            f.write(f"사용 가능한 충전소: {stats['available_ev_spots']}개\n\n")
            
            f.write("=== 통계 ===\n")
            f.write(f"총 입차 시도: {stats['total_entries']}회\n")
            f.write(f"성공한 주차: {stats['successful_parks']}회\n")
            f.write(f"실패한 주차: {stats['failed_parks']}회\n")
            f.write(f"총 충전 시도: {stats['total_charges']}회\n")
            f.write(f"실패한 충전 시도: {charge_fail_count}회\n")
            f.write(f"출차: {depart_count}회\n\n")
            
            f.write("==== 최적화/운영 지표 ====\n")
            if stats['total_charger_spots'] > 0:
                charger_cost = self.logger.calculate_charger_cost(stats['total_charger_spots'])
                idle_rate = self.logger.calculate_charger_idle_rate(self.env.now, stats['total_charger_spots'])
                charge_fail_rate = self.logger.calculate_charge_fail_rate()
                parking_fail_rate = self.logger.calculate_parking_fail_rate()
                
                f.write(f"충전소 설치+유지 총비용: {charger_cost:,} 원\n")
                f.write(f"충전소 공실률: {idle_rate * 100:.2f} %\n")
                f.write(f"충전 실패율: {charge_fail_rate * 100:.2f} %\n")
                f.write(f"주차 실패율: {parking_fail_rate * 100:.2f} %\n")
            else:
                f.write("충전소가 설치되어 있지 않습니다.\n")
                f.write(f"주차 실패율: {self.logger.calculate_parking_fail_rate() * 100:.2f} %\n")
        
        print(f"\n[INFO] 시뮬레이션 결과가 {filename}에 저장되었습니다.")

    def print_summary(self, output_dir: str = None) -> None:
        """
        시뮬레이션 결과 요약을 출력하고 파일로 저장
        
        Args:
            output_dir: 결과를 저장할 디렉토리 경로 (선택적)
        """
        stats = self.get_statistics()
        
        # 시뮬레이션 진행 시간을 일/시간/분/초로 변환
        def format_time(seconds):
            days = int(seconds // 86400)
            hours = int((seconds % 86400) // 3600)
            minutes = int((seconds % 3600) // 60)
            secs = int(seconds % 60)
            return f"{days}일 {hours}시간 {minutes}분 {secs}초"
        
        sim_time_str = format_time(self.env.now)
        
        # 출차 횟수 계산
        df = self.logger.get_dataframe()
        depart_count = len(df[df.event == "depart"]) if not df.empty else 0
        
        # 실패한 충전 시도 계산
        charge_fail_count = len(df[df.event == "charge_fail"]) if not df.empty else 0
        
        print("\n==== 시뮬레이션 결과 ====")
        print(f"총 진행 시간: {sim_time_str}")
        
        print("\n==== 주차장 세팅 ====")
        print(f"전체 주차면: {stats['total_parking_spots']}면 (일반 주차면)")
        print(f"전체 충전소: {stats['total_charger_spots']}개")
        
        print("\n=== 주차장 상태 (현재) ===")
        print(f"({sim_time_str}) 동안 진행 후 현재 상황")
        print(f"총 주차된 차량: {stats['total_parked']}대")
        print(f"이중주차 차량: {stats['double_parked']}대")
        print(f"사용 가능한 주차면: {stats['available_spots']}면")
        print(f"사용 가능한 충전소: {stats['available_ev_spots']}개")
        
        print("\n=== 통계 ===")
        print(f"총 입차 시도: {stats['total_entries']}회")
        print(f"성공한 주차: {stats['successful_parks']}회")
        print(f"실패한 주차: {stats['failed_parks']}회")
        print(f"총 충전 시도: {stats['total_charges']}회")
        print(f"실패한 충전 시도: {charge_fail_count}회")
        print(f"출차: {depart_count}회")
        
        # 충전소 관련 지표
        if stats['total_charger_spots'] > 0:
            charger_cost = self.logger.calculate_charger_cost(stats['total_charger_spots'])
            idle_rate = self.logger.calculate_charger_idle_rate(self.env.now, stats['total_charger_spots'])
            charge_fail_rate = self.logger.calculate_charge_fail_rate()
            parking_fail_rate = self.logger.calculate_parking_fail_rate()
            
            print("\n==== 최적화/운영 지표 ====")
            print(f"충전소 설치+유지 총비용: {charger_cost:,} 원")
            print(f"충전소 공실률: {idle_rate * 100:.2f} %")
            print(f"충전 실패율: {charge_fail_rate * 100:.2f} %")
            print(f"주차 실패율: {parking_fail_rate * 100:.2f} %")
        else:
            print("\n==== 최적화/운영 지표 ====")
            print("충전소가 설치되어 있지 않습니다.")
            print(f"주차 실패율: {self.logger.calculate_parking_fail_rate() * 100:.2f} %")
        
        # output_dir이 제공된 경우에만 파일로 저장
        if output_dir:
            self.save_summary_to_file(output_dir)

    def generate_plots(self) -> None:
        """시뮬레이션 결과를 시각화하는 그래프를 생성"""
        self.logger.generate_plots() 