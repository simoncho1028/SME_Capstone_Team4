"""
주차장 시뮬레이션을 실행하는 모듈
"""
import simpy
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import os
import random
import numpy as np
import csv

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
                 ev_count: int = None,
                 results_dir: str = None):
        """
        시뮬레이션 객체를 초기화합니다.
        
        Args:
            env: SimPy 환경
            parking_manager: 주차장 관리자
            logger: 이벤트 로깅을 위한 로거 객체
            total_vehicle_count: 전체 차량 수 (CLI에서 지정된 --normal + --ev 값)
            normal_count: 일반 차량 수
            ev_count: EV 차량 수
            results_dir: 결과 저장 디렉토리 (기본값: None, logger의 log_file이 있는 디렉토리 사용)
        """
        self.env = env
        self.parking_manager = parking_manager
        self.logger = logger
        self.total_vehicle_count = total_vehicle_count
        self.normal_count = normal_count
        self.ev_count = ev_count
        
        # results_dir가 지정되지 않은 경우 logger의 log_file이 있는 디렉토리 사용
        if results_dir is None:
            self.results_dir = os.path.dirname(logger.log_file)
        else:
            self.results_dir = results_dir
        
        # results_dir가 없으면 생성
        os.makedirs(self.results_dir, exist_ok=True)
        
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
        
        self.failed_vehicles_today = set()  # park_fail 차량 관리

    def log_event(self, vehicle: Vehicle, event: str, results_dir: str = None) -> None:
        """이벤트 로깅"""
        pos = None
        floor = None
        if event in ["park_success", "park_fail", "charge_start", "charge_complete"]:
            vehicle_pos = self.parking_manager.get_vehicle_position(vehicle.vehicle_id)
            if vehicle_pos:
                floor = vehicle_pos[0]  # 층 정보
                pos = (vehicle_pos[1], vehicle_pos[2])  # (row, col)
            
            # 주차 성공 시 거리 계산 및 기록
            if event == "park_success" and pos is not None and results_dir is not None:
                # 차량의 동 좌표
                building_coords = self.parking_manager.parking_system.building_coordinates
                bx, by = building_coords.get(vehicle.building_id, (None, None))
                if bx is not None and by is not None:
                    distance = abs(bx - pos[0]) + abs(by - pos[1])
                    distance_path = os.path.join(results_dir, "distance_log.csv")
                    write_header = not os.path.exists(distance_path)
                    with open(distance_path, "a", encoding="utf-8") as f:
                        if write_header:
                            f.write("no,type,distance\n")
                        f.write(f"{vehicle.vehicle_id},{vehicle.vehicle_type},{distance}\n")
        
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
        self.log_event(vehicle, "arrive", self.results_dir)
        
        # 차량을 active_vehicles에 추가
        self.active_vehicles[vehicle.vehicle_id] = vehicle
        
        # park_vehicle의 반환값: (주차성공, charge_fail)
        park_success, charge_fail = self.parking_manager.park_vehicle(vehicle)
        if charge_fail:
            # charge_fail 이벤트 로깅 추가
            self.log_event(vehicle, "charge_fail", self.results_dir)
            # 통계에 반영
            if "charge_fail" not in self.stats:
                self.stats["charge_fail"] = 0
            self.stats["charge_fail"] += 1
            self.failed_vehicles_today.add(vehicle.vehicle_id)  # park_fail 차량 기록
        if park_success:
            self.stats["successful_parks"] += 1
            self.log_event(vehicle, "park_success", self.results_dir)
            
            # 주차 성공 시 출차 프로세스 시작
            self.env.process(self.vehicle_exit_process(vehicle))
            
            # 전기차이고 충전이 필요하며 충전소에 주차된 경우에만 충전 시작
            if (vehicle.vehicle_type == "ev" and 
                vehicle.needs_charging() and 
                self.parking_manager.get_vehicle_position(vehicle.vehicle_id) in self.parking_manager.ev_chargers):
                vehicle.start_charging(self.env.now)
                self.stats["total_charges"] += 1
                self.log_event(vehicle, "charge_start", self.results_dir)
                # 충전 완료 예약 프로세스 추가
                self.env.process(self.ev_charge_complete_process(vehicle))
        else:
            self.stats["failed_parks"] += 1
            self.log_event(vehicle, "park_fail", self.results_dir)
            self.failed_vehicles_today.add(vehicle.vehicle_id)
            # 반드시 대기열에서도 제거
            if vehicle.vehicle_id in self.outside_vehicles:
                self.outside_vehicles.pop(vehicle.vehicle_id)

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
        from src.config import normalized_entry_ratios, ENTRY_RATIO_MEAN, ENTRY_RATIO_STD, ENTRY_RATIO_MIN, ENTRY_RATIO_MAX
        
        # 하루의 시작 시간 계산 (현재 시간의 0시)
        day_start = current_time - (current_time % 86400)
        
        # --- 전체 차량 pool에서 EV/normal 분리 ---
        all_vehicles = list(self.active_vehicles.values()) + list(self.outside_vehicles.values())
        ev_vehicles = [v for v in all_vehicles if v.vehicle_type == "ev"]
        normal_vehicles = [v for v in all_vehicles if v.vehicle_type != "ev"]

        # 이미 주차 중인 차량은 제외 (active_vehicles에 있는 차량은 제외)
        ev_candidates = [v for v in ev_vehicles if v.vehicle_id not in self.active_vehicles and v.vehicle_id not in self.failed_vehicles_today]
        normal_candidates = [v for v in normal_vehicles if v.vehicle_id not in self.active_vehicles and v.vehicle_id not in self.failed_vehicles_today]

        # 매일 입차 비율 설정 (정규분포 사용)
        ev_ratio = np.random.normal(ENTRY_RATIO_MEAN, ENTRY_RATIO_STD)
        normal_ratio = np.random.normal(ENTRY_RATIO_MEAN, ENTRY_RATIO_STD)
        
        # 비율을 설정된 범위로 제한
        ev_ratio = max(ENTRY_RATIO_MIN, min(ENTRY_RATIO_MAX, ev_ratio))
        normal_ratio = max(ENTRY_RATIO_MIN, min(ENTRY_RATIO_MAX, normal_ratio))
        
        ev_entries = int(len(ev_candidates) * ev_ratio)
        normal_entries = int(len(normal_candidates) * normal_ratio)

        # 랜덤 샘플링 (중복 허용)
        ev_today = random.choices(ev_candidates, k=ev_entries)
        normal_today = random.choices(normal_candidates, k=normal_entries)
        today_vehicles = ev_today + normal_today
        random.shuffle(today_vehicles)

        # 시간대별 입차량 계산 (전체 입차 차량 수 기준)
        total_entries = len(today_vehicles)
        hourly_entries = []
        for ratio in normalized_entry_ratios:
            # ±10% 노이즈 추가
            noise = np.random.uniform(0.9, 1.1)
            entries = int(total_entries * ratio * noise)
            hourly_entries.append(entries)
        
        # 각 시간대별로 차량 입차 시간 샘플링
        self.daily_entry_plan.clear()
        idx = 0
        for hour, entries in enumerate(hourly_entries):
            if entries > 0:
                hour_start = day_start + (hour * 3600)
                entry_times = np.random.uniform(hour_start, hour_start + 3600, entries)
                for entry_time in sorted(entry_times):
                    if idx < len(today_vehicles):
                        vehicle = today_vehicles[idx]
                        # outside_vehicles에서 제거 (입차 대기열에서만 제거)
                        if vehicle.vehicle_id in self.outside_vehicles:
                            self.outside_vehicles.pop(vehicle.vehicle_id)
                        self.daily_entry_plan.append((entry_time, vehicle))
                        idx += 1
    
    def daily_planning_process(self):
        """매일 0시마다 새로운 입차 계획을 수립하는 프로세스"""
        while True:
            # 현재 시간의 다음 날 0시까지 대기
            next_day = (self.env.now - (self.env.now % 86400)) + 86400
            yield self.env.timeout(next_day - self.env.now)
            
            self.failed_vehicles_today.clear()  # 하루가 끝나면 실패 차량 초기화
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
        
        # 입차 프로세스 시작
        self.env.process(self.vehicle_entry_process())
        
        # 시뮬레이션 실행
        self.env.run(until=until)
        
        # 결과 저장
        self.save_summary_to_file(self.results_dir)

    def get_statistics(self) -> Dict:
        """시뮬레이션 통계 반환"""
        stats = self.stats.copy()
        stats.update(self.parking_manager.get_parking_status())
        return stats

    def calculate_metrics(self, output_dir: str) -> dict:
        """
        시뮬레이션 지표를 계산합니다.
        
        Args:
            output_dir: 결과 저장 디렉토리
            
        Returns:
            dict: 계산된 지표들
        """
        metrics = {
            "charger_count": self.parking_manager.total_charger_spots,
            "charge_fail_rate": 0.0,
            "parking_fail_rate": 0.0,
            "avg_normal_distance": 0.0,
            "avg_ev_distance": 0.0
        }
        
        # 충전 실패율 계산
        total_charge_attempts = self.stats.get("total_charges", 0)
        charge_fails = self.stats.get("charge_fail", 0)
        if total_charge_attempts > 0:
            metrics["charge_fail_rate"] = (charge_fails / total_charge_attempts) * 100
        
        # 주차 실패율 계산
        total_entries = self.stats.get("total_entries", 0)
        failed_parks = self.stats.get("failed_parks", 0)
        if total_entries > 0:
            metrics["parking_fail_rate"] = (failed_parks / total_entries) * 100
        
        # 거리 통계 계산
        distance_path = os.path.join(output_dir, "distance_log.csv")
        try:
            normal_dists = []
            ev_dists = []
            with open(distance_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row["type"] == "normal":
                        normal_dists.append(float(row["distance"]))
                    elif row["type"] == "ev":
                        ev_dists.append(float(row["distance"]))
            
            if normal_dists:
                metrics["avg_normal_distance"] = sum(normal_dists) / len(normal_dists)
            if ev_dists:
                metrics["avg_ev_distance"] = sum(ev_dists) / len(ev_dists)
        except Exception as e:
            print(f"[WARN] 거리 통계 계산 중 오류 발생: {e}")
        
        return metrics

    def save_summary_to_file(self, output_dir: str) -> None:
        """
        시뮬레이션 결과 요약을 파일로 저장합니다.
        
        Args:
            output_dir: 저장할 디렉토리 경로
        """
        # 지표 계산
        metrics = self.calculate_metrics(output_dir)
        
        # 시뮬레이션 진행 시간을 일/시간/분/초로 변환
        def format_time(seconds):
            days = int(seconds // 86400)
            hours = int((seconds % 86400) // 3600)
            minutes = int((seconds % 3600) // 60)
            secs = int(seconds % 60)
            return f"{days}일 {hours}시간 {minutes}분 {secs}초"
        
        sim_time_str = format_time(self.env.now)
        
        # 주차장 상태 계산
        stats = self.get_statistics()
        total_spots = stats['total_parking_spots'] + stats['total_charger_spots']
        total_parked = stats['total_parked']
        occupied_normal = sum(1 for v in self.parking_manager.parked_vehicles.values() if v not in self.parking_manager.ev_chargers)
        occupied_charger = sum(1 for v in self.parking_manager.parked_vehicles.values() if v in self.parking_manager.ev_chargers)
        available_normal = stats['total_parking_spots'] - occupied_normal
        available_charger = stats['total_charger_spots'] - occupied_charger
        
        # 성공률 계산
        success_rate = (stats['successful_parks'] / stats['total_entries'] * 100) if stats['total_entries'] > 0 else 0
        fail_rate = (stats['failed_parks'] / stats['total_entries'] * 100) if stats['total_entries'] > 0 else 0
        
        # EV 충전 통계 집계
        df = self.logger.get_dataframe()
        ev_df = df[df['type'] == 'ev']
        charge_starts = ev_df[ev_df['event'] == 'charge_start']
        charge_fails = ev_df[ev_df['event'] == 'charge_fail']
        
        charge_try = len(charge_starts) + len(charge_fails)  # 실제 충전 시도 횟수
        charge_success = len(charge_starts)  # 충전 시작된 횟수
        charge_fail = len(charge_fails)  # 충전 실패 횟수
        
        # 출차 횟수 계산
        depart_count = len(df[df.event == "depart"]) if not df.empty else 0
        
        # 평균 주차 시간 계산
        park_events = df[df.event.isin(["park_start", "depart"])].copy()
        park_events = park_events.sort_values(["id", "time"])
        total_parking_time = 0
        parking_count = 0
        
        # 각 차량별로 주차 시작과 출차 이벤트를 매칭
        for vehicle_id in park_events.id.unique():
            v_events = park_events[park_events.id == vehicle_id].sort_values("time")
            park_starts = v_events[v_events.event == "park_start"]
            departs = v_events[v_events.event == "depart"]
            
            # 주차 시작과 출차 이벤트를 순서대로 매칭
            for i in range(min(len(park_starts), len(departs))):
                start_time = park_starts.iloc[i].time
                end_time = departs.iloc[i].time
                if end_time > start_time:  # 유효한 주차 시간인 경우만
                    parking_duration = end_time - start_time
                    total_parking_time += parking_duration
                    parking_count += 1
        
        avg_parking_time = total_parking_time / parking_count if parking_count > 0 else 0
        avg_parking_hours = avg_parking_time / 3600  # 초를 시간으로 변환
        
        # 결과 문자열 생성
        result_str = f"""
=== 시뮬레이션 결과 ===
==== 시뮬레이션 결과 요약 ====

[총 진행 시간]
- {sim_time_str}

[주차장 세팅]
- 전체 주차면: {total_spots}면 (일반 {stats['total_parking_spots']} + 충전소 {stats['total_charger_spots']})
- 충전소 수: {stats['total_charger_spots']}개
- 일반 차량: {self.normal_count if self.normal_count is not None else 'N/A'}대
- 전기차(EV): {self.ev_count if self.ev_count is not None else 'N/A'}대

[현재 주차장 상태]
- 전체 주차 차량: {total_parked} / {total_spots}
  - 일반 주차면 점유: {occupied_normal} / {stats['total_parking_spots']}
  - 충전소 점유: {occupied_charger} / {stats['total_charger_spots']}
- 사용 가능 일반 주차면: {available_normal}
- 사용 가능 충전소: {available_charger}

[이용 통계]
- 총 입차 시도: {stats['total_entries']}회
  - 주차 성공: {stats['successful_parks']}회 (성공률: {success_rate:.1f}%)
  - 주차 실패: {stats['failed_parks']}회 (실패율: {fail_rate:.1f}%)
- 총 출차: {depart_count}회
- 평균 주차 시간: {avg_parking_hours:.1f}시간
- EV 충전 시도: {charge_try}회
  - 충전 성공: {charge_success}회 (성공률: {(charge_success/charge_try*100) if charge_try else 0:.1f}%)
  - 충전 실패: {charge_fail}회 (실패율: {(charge_fail/charge_try*100) if charge_try else 0:.1f}%)

[지표]
- 충전소 개수: {metrics['charger_count']}개
- 충전 실패율: {metrics['charge_fail_rate']:.1f}%
- 주차 실패율: {metrics['parking_fail_rate']:.1f}%
- 평균 distance (일반 차량): {metrics['avg_normal_distance']:.1f}
- 평균 distance (EV): {metrics['avg_ev_distance']:.1f}
"""
        
        # 결과 저장 (simulation_summary.txt 파일에 저장)
        summary_path = os.path.join(output_dir, "simulation_summary.txt")
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write(result_str)
        
        # 터미널에 출력
        print(result_str)

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
        
        # 평균 주차 시간 계산
        park_events = df[df.event.isin(["park_start", "depart"])].copy()
        park_events = park_events.sort_values(["id", "time"])
        total_parking_time = 0
        parking_count = 0
        
        print("\n[디버깅] 주차 이벤트 분석:")
        print(f"총 이벤트 수: {len(park_events)}")
        print(f"주차 시작 이벤트 수: {len(park_events[park_events.event == 'park_start'])}")
        print(f"출차 이벤트 수: {len(park_events[park_events.event == 'depart'])}")
        
        # 각 차량별로 주차 시작과 출차 이벤트를 매칭
        for vehicle_id in park_events.id.unique():
            v_events = park_events[park_events.id == vehicle_id].sort_values("time")
            park_starts = v_events[v_events.event == "park_start"]
            departs = v_events[v_events.event == "depart"]
            
            print(f"\n차량 {vehicle_id}:")
            print(f"주차 시작 이벤트: {len(park_starts)}개")
            print(f"출차 이벤트: {len(departs)}개")
            
            # 주차 시작과 출차 이벤트를 순서대로 매칭
            for i in range(min(len(park_starts), len(departs))):
                start_time = park_starts.iloc[i].time
                end_time = departs.iloc[i].time
                if end_time > start_time:  # 유효한 주차 시간인 경우만
                    parking_duration = end_time - start_time
                    total_parking_time += parking_duration
                    parking_count += 1
                    print(f"주차 시간: {parking_duration/3600:.2f}시간 (시작: {start_time/3600:.2f}시간, 종료: {end_time/3600:.2f}시간)")
        
        print(f"\n총 주차 시간: {total_parking_time/3600:.2f}시간")
        print(f"주차 횟수: {parking_count}회")
        
        avg_parking_time = total_parking_time / parking_count if parking_count > 0 else 0
        avg_parking_hours = avg_parking_time / 3600  # 초를 시간으로 변환
        
        # 주차장 상태 계산
        total_spots = stats['total_parking_spots'] + stats['total_charger_spots']
        total_parked = stats['total_parked']
        occupied_normal = sum(1 for v in self.parking_manager.parked_vehicles.values() if v not in self.parking_manager.ev_chargers)
        occupied_charger = sum(1 for v in self.parking_manager.parked_vehicles.values() if v in self.parking_manager.ev_chargers)
        available_normal = stats['total_parking_spots'] - occupied_normal
        available_charger = stats['total_charger_spots'] - occupied_charger
        
        # 성공률 계산
        success_rate = (stats['successful_parks'] / stats['total_entries'] * 100) if stats['total_entries'] > 0 else 0
        fail_rate = (stats['failed_parks'] / stats['total_entries'] * 100) if stats['total_entries'] > 0 else 0
        
        # EV 충전 통계 집계
        ev_df = df[df['type'] == 'ev']
        charge_starts = ev_df[ev_df['event'] == 'charge_start']
        charge_fails = ev_df[ev_df['event'] == 'charge_fail']
        
        charge_try = len(charge_starts) + len(charge_fails)  # 실제 충전 시도 횟수
        charge_success = len(charge_starts)  # 충전 시작된 횟수
        charge_fail = len(charge_fails)  # 충전 실패 횟수
        
        # 결과 문자열 생성
        result_str = f"""
=== 시뮬레이션 결과 ===
==== 시뮬레이션 결과 요약 ====

[총 진행 시간]
- {sim_time_str}

[주차장 세팅]
- 전체 주차면: {total_spots}면 (일반 {stats['total_parking_spots']} + 충전소 {stats['total_charger_spots']})
- 충전소 수: {stats['total_charger_spots']}개
- 일반 차량: {self.normal_count if self.normal_count is not None else 'N/A'}대
- 전기차(EV): {self.ev_count if self.ev_count is not None else 'N/A'}대

[현재 주차장 상태]
- 전체 주차 차량: {total_parked} / {total_spots}
  - 일반 주차면 점유: {occupied_normal} / {stats['total_parking_spots']}
  - 충전소 점유: {occupied_charger} / {stats['total_charger_spots']}
- 사용 가능 일반 주차면: {available_normal}
- 사용 가능 충전소: {available_charger}

[이용 통계]
- 총 입차 시도: {stats['total_entries']}회
  - 주차 성공: {stats['successful_parks']}회 (성공률: {success_rate:.1f}%)
  - 주차 실패: {stats['failed_parks']}회 (실패율: {fail_rate:.1f}%)
- 총 출차: {depart_count}회
- 평균 주차 시간: {avg_parking_hours:.1f}시간
- EV 충전 시도: {charge_try}회
  - 충전 성공: {charge_success}회 (성공률: {(charge_success/charge_try*100) if charge_try else 0:.1f}%)
  - 충전 실패: {charge_fail}회 (실패율: {(charge_fail/charge_try*100) if charge_try else 0:.1f}%)
"""
        
        # 결과 출력
        print("\n".join(result_str.splitlines()))
        
        # output_dir이 제공된 경우에만 파일로 저장
        if output_dir:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = os.path.join(output_dir, f"simulation_summary_{timestamp}.txt")
            with open(filename, "w", encoding="utf-8") as f:
                f.write("\n".join(result_str.splitlines()))
            print(f"\n[INFO] 시뮬레이션 결과가 {filename}에 저장되었습니다.")

    def generate_plots(self) -> None:
        """시뮬레이션 결과를 시각화하는 그래프를 생성"""
        self.logger.generate_plots() 