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
                 total_vehicle_count: int = 866,
                 normal_count: int = None,
                 ev_count: int = None):
        """
        시뮬레이션 객체를 초기화합니다.
        
        Args:
            env: SimPy 환경
            parking_manager: 주차장 관리자
            logger: 이벤트 로깅을 위한 로거 객체
            total_vehicle_count: 전체 차량 수 (CLI에서 지정된 --normal + --ev 값)
            normal_count: 일반 차량 수
            ev_count: EV 차량 수
        """
        self.env = env
        self.parking_manager = parking_manager
        self.logger = logger
        self.total_vehicle_count = total_vehicle_count
        self.normal_count = normal_count
        self.ev_count = ev_count
        
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
        
        # park_vehicle의 반환값: (주차성공, charge_fail)
        park_success, charge_fail = self.parking_manager.park_vehicle(vehicle)
        if charge_fail:
            # log 기록 없이 통계에만 반영
            if "charge_fail" not in self.stats:
                self.stats["charge_fail"] = 0
            self.stats["charge_fail"] += 1
        if park_success:
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
                # 충전 완료 예약 프로세스 추가
                self.env.process(self.ev_charge_complete_process(vehicle))
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

    def ev_charge_complete_process(self, vehicle: Vehicle):
        """
        EV 충전 완료 예약 프로세스 (충전 시작 후 assigned_charging_time 경과 시 charge_complete 로그)
        """
        if vehicle.assigned_charging_time is not None:
            yield self.env.timeout(vehicle.assigned_charging_time * 60)  # 분 -> 초
            self.log_event(vehicle, "charge_complete")

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
            f.write("==== 시뮬레이션 결과 요약 ====\n")
            f.write(f"\n[총 진행 시간]\n")
            f.write(f"- {sim_time_str}\n")
            
            f.write(f"\n[주차장 세팅]\n")
            f.write(f"- 전체 주차면: {stats['total_parking_spots'] + stats['total_charger_spots']}면 (일반 {stats['total_parking_spots']} + 충전소 {stats['total_charger_spots']})\n")
            f.write(f"- 충전소 수: {stats['total_charger_spots']}개\n")
            f.write(f"- 일반 차량: {self.normal_count if self.normal_count is not None else 'N/A'}대\n")
            f.write(f"- 전기차(EV): {self.ev_count if self.ev_count is not None else 'N/A'}대\n")
            
            f.write(f"\n[현재 주차장 상태]\n")
            total_spots = stats['total_parking_spots'] + stats['total_charger_spots']
            total_parked = stats['total_parked']
            occupied_normal = sum(1 for v in self.parking_manager.parked_vehicles.values() if v not in self.parking_manager.ev_chargers)
            occupied_charger = sum(1 for v in self.parking_manager.parked_vehicles.values() if v in self.parking_manager.ev_chargers)
            available_normal = stats['total_parking_spots'] - occupied_normal
            available_charger = stats['total_charger_spots'] - occupied_charger

            f.write(f"- 전체 주차 차량: {total_parked} / {total_spots}\n")
            f.write(f"  - 일반 주차면 점유: {occupied_normal} / {stats['total_parking_spots']}\n")
            f.write(f"  - 충전소 점유: {occupied_charger} / {stats['total_charger_spots']}\n")
            f.write(f"- 사용 가능 일반 주차면: {available_normal}\n")
            f.write(f"- 사용 가능 충전소: {available_charger}\n")
            
            f.write(f"\n[이용 통계]\n")
            f.write(f"- 총 입차 시도: {stats['total_entries']}회\n")
            success_rate = (stats['successful_parks'] / stats['total_entries'] * 100) if stats['total_entries'] > 0 else 0
            fail_rate = (stats['failed_parks'] / stats['total_entries'] * 100) if stats['total_entries'] > 0 else 0
            f.write(f"  - 주차 성공: {stats['successful_parks']}회 (성공률: {success_rate:.1f}%)\n")
            f.write(f"  - 주차 실패: {stats['failed_parks']}회 (실패율: {fail_rate:.1f}%)\n")
            f.write(f"- 총 출차: {depart_count}회\n")
 
            # --- EV 충전 통계 집계 ---
            ev_arrive = df[(df['type'] == 'ev') & (df['event'] == 'arrive') & (df['battery'] < 80)]
            ev_ids = set(ev_arrive['id'])
            charge_success = 0
            charge_fail = 0
            for vid in ev_ids:
                sub = df[df['id'] == vid]
                if (sub['event'] == 'charge_start').any():
                    charge_success += 1
                elif (sub['event'] == 'park_success').any():
                    charge_fail += 1
            charge_try = len(ev_ids)

            f.write(f"- EV 충전 시도: {charge_try}회\n")
            f.write(f"  - 충전 성공: {charge_success}회 (성공률: {(charge_success/charge_try*100) if charge_try else 0:.1f}%)\n")
            f.write(f"  - 충전 실패: {charge_fail}회 (실패율: {(charge_fail/charge_try*100) if charge_try else 0:.1f}%)\n")
            
    
        
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
        
        print("\n==== 시뮬레이션 결과 요약 ====")
        print(f"\n[총 진행 시간]")
        print(f"- {sim_time_str}")
        
        print(f"\n[주차장 세팅]")
        print(f"- 전체 주차면: {stats['total_parking_spots'] + stats['total_charger_spots']}면 (일반 {stats['total_parking_spots']} + 충전소 {stats['total_charger_spots']})")
        print(f"- 충전소 수: {stats['total_charger_spots']}개")
        print(f"- 일반 차량: {self.normal_count if self.normal_count is not None else 'N/A'}대")
        print(f"- 전기차(EV): {self.ev_count if self.ev_count is not None else 'N/A'}대")
        
        print(f"\n[현재 주차장 상태]")
        total_spots = stats['total_parking_spots'] + stats['total_charger_spots']
        total_parked = stats['total_parked']
        occupied_normal = sum(1 for v in self.parking_manager.parked_vehicles.values() if v not in self.parking_manager.ev_chargers)
        occupied_charger = sum(1 for v in self.parking_manager.parked_vehicles.values() if v in self.parking_manager.ev_chargers)
        available_normal = stats['total_parking_spots'] - occupied_normal
        available_charger = stats['total_charger_spots'] - occupied_charger

        print(f"- 전체 주차 차량: {total_parked} / {total_spots}")
        print(f"  - 일반 주차면 점유: {occupied_normal} / {stats['total_parking_spots']}")
        print(f"  - 충전소 점유: {occupied_charger} / {stats['total_charger_spots']}")
        print(f"- 사용 가능 일반 주차면: {available_normal}")
        print(f"- 사용 가능 충전소: {available_charger}")
        
        print(f"\n[이용 통계]")
        print(f"- 총 입차 시도: {stats['total_entries']}회")
        success_rate = (stats['successful_parks'] / stats['total_entries'] * 100) if stats['total_entries'] > 0 else 0
        fail_rate = (stats['failed_parks'] / stats['total_entries'] * 100) if stats['total_entries'] > 0 else 0
        print(f"  - 주차 성공: {stats['successful_parks']}회 (성공률: {success_rate:.1f}%)")
        print(f"  - 주차 실패: {stats['failed_parks']}회 (실패율: {fail_rate:.1f}%)")
        print(f"- 총 출차: {depart_count}회")

        # --- EV 충전 통계 집계 ---
        ev_arrive = df[(df['type'] == 'ev') & (df['event'] == 'arrive') & (df['battery'] < 80)]
        ev_ids = set(ev_arrive['id'])
        charge_success = 0
        charge_fail = 0
        for vid in ev_ids:
            sub = df[df['id'] == vid]
            if (sub['event'] == 'charge_start').any():
                charge_success += 1
            elif (sub['event'] == 'park_success').any():
                charge_fail += 1
        charge_try = len(ev_ids)

        print(f"- EV 충전 시도: {charge_try}회")
        print(f"  - 충전 성공: {charge_success}회 (성공률: {(charge_success/charge_try*100) if charge_try else 0:.1f}%)")
        print(f"  - 충전 실패: {charge_fail}회 (실패율: {(charge_fail/charge_try*100) if charge_try else 0:.1f}%)")
        
       
        
        # output_dir이 제공된 경우에만 파일로 저장
        if output_dir:
            self.save_summary_to_file(output_dir)

    def generate_plots(self) -> None:
        """시뮬레이션 결과를 시각화하는 그래프를 생성"""
        self.logger.generate_plots() 