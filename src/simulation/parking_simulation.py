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
                 ratio_min: float = 1.0,
                 ratio_max: float = 1.3):
        """
        시뮬레이션 객체를 초기화합니다.
        
        Args:
            env: SimPy 환경
            parking_manager: 주차장 관리자
            logger: 이벤트 로깅을 위한 로거 객체
            total_vehicle_count: 전체 차량 수 (CLI에서 지정된 --normal + --ev 값)
            normal_count: 일반 차량 수
            ev_count: EV 차량 수
            ratio_min: 입차 비율의 최소값
            ratio_max: 입차 비율의 최대값
        """
        self.env = env
        self.parking_manager = parking_manager
        self.logger = logger
        self.total_vehicle_count = total_vehicle_count
        self.normal_count = normal_count
        self.ev_count = ev_count
        self.ratio_min = ratio_min
        self.ratio_max = ratio_max
        
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
        self.daily_entry_plan: List[float] = []

    def log_event(self, vehicle: Vehicle, event: str, results_dir: str = None) -> None:
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
        # park_success 시 이동 거리 기록
        if event == "park_success" and pos is not None:
            building_coords = self.parking_manager.parking_system.building_coordinates
            bx, by = building_coords.get(vehicle.building_id, (None, None))
            if bx is not None and by is not None:
                distance = abs(bx - pos[0]) + abs(by - pos[1])
                # results_dir이 None이면 로거의 log_file이 있는 디렉토리 사용
                if results_dir is None:
                    results_dir = os.path.dirname(self.logger.log_file)
                distance_path = os.path.join(results_dir, "distance_log.csv")
                write_header = not os.path.exists(distance_path)
                with open(distance_path, "a", encoding="utf-8") as f:
                    if write_header:
                        f.write("id,type,distance\n")
                    f.write(f"{vehicle.vehicle_id},{vehicle.vehicle_type},{distance}\n")

    def handle_vehicle_entry(self, vehicle: Vehicle, results_dir: str = None) -> None:
        """
        차량 입차 처리
        Args:
            vehicle: 입차하는 차량
            results_dir: 결과 저장 디렉토리 경로
        """
        self.stats["total_entries"] += 1
        self.log_event(vehicle, "arrive", results_dir)
        # 차량을 active_vehicles에 추가
        self.active_vehicles[vehicle.vehicle_id] = vehicle
        # park_vehicle의 반환값: (주차성공, charge_fail)
        park_success, charge_fail = self.parking_manager.park_vehicle(vehicle)
        if charge_fail:
            self.log_event(vehicle, "charge_fail", results_dir)
            if "charge_fail" not in self.stats:
                self.stats["charge_fail"] = 0
            self.stats["charge_fail"] += 1
        if park_success:
            self.stats["successful_parks"] += 1
            self.log_event(vehicle, "park_success", results_dir)
            self.env.process(self.vehicle_exit_process(vehicle))
            if (vehicle.vehicle_type == "ev" and 
                vehicle.needs_charging() and 
                self.parking_manager.get_vehicle_position(vehicle.vehicle_id) in self.parking_manager.ev_chargers):
                vehicle.start_charging(self.env.now)
                self.stats["total_charges"] += 1
                self.log_event(vehicle, "charge_start", results_dir)
                self.env.process(self.ev_charge_complete_process(vehicle))
        else:
            self.stats["failed_parks"] += 1
            self.log_event(vehicle, "park_fail", results_dir)

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
        # 입차할 총 차량 수 계산 (outside_vehicles 기준)
        total_vehicles = len(self.outside_vehicles)
        # 시간대별 입차 비율로 입차 시간만 생성
        hourly_entries = []
        for ratio in normalized_entry_ratios:
            noise = np.random.uniform(self.ratio_min, self.ratio_max)
            entries = int(total_vehicles * ratio * noise)
            hourly_entries.append(entries)
        # 각 시간대별로 입차 시간 샘플링
        self.daily_entry_plan.clear()
        for hour, entries in enumerate(hourly_entries):
            if entries > 0:
                hour_start = day_start + (hour * 3600)
                entry_times = np.random.uniform(hour_start, hour_start + 3600, entries)
                for entry_time in sorted(entry_times):
                    self.daily_entry_plan.append(entry_time)
        # 입차 시간 정렬
        self.daily_entry_plan.sort()

    def daily_planning_process(self):
        """매일 0시마다 새로운 입차 계획을 수립하는 프로세스"""
        while True:
            # 현재 시간의 다음 날 0시까지 대기
            next_day = (self.env.now - (self.env.now % 86400)) + 86400
            yield self.env.timeout(next_day - self.env.now)
            
            # 새로운 하루의 입차 계획 수립
            self.plan_daily_entries(self.env.now)
    
    def vehicle_entry_process(self, results_dir: str = None):
        """
        입차 계획에 따라 차량을 입차시키는 프로세스
        """
        while True:
            if self.daily_entry_plan:
                next_entry_time = self.daily_entry_plan[0]
                if next_entry_time > self.env.now:
                    yield self.env.timeout(next_entry_time - self.env.now)
                if self.outside_vehicles:
                    vehicle_id, vehicle = random.choice(list(self.outside_vehicles.items()))
                    self.outside_vehicles.pop(vehicle_id)
                    self.handle_vehicle_entry(vehicle, results_dir)
                self.daily_entry_plan.pop(0)
            else:
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

    def run(self, until: float, results_dir: str = None) -> None:
        """
        시뮬레이션 실행
        Args:
            until: 시뮬레이션 종료 시각 (초)
            results_dir: 결과 저장 디렉토리 경로
        """
        self.plan_daily_entries(self.env.now)
        self.env.process(self.daily_planning_process())
        self.env.process(self.vehicle_entry_process(results_dir))
        self.env.run(until=until)

    def get_statistics(self) -> Dict:
        """시뮬레이션 통계 반환"""
        stats = self.stats.copy()
        stats.update(self.parking_manager.get_parking_status())
        return stats

    def save_summary_to_file(self, output_dir: str) -> None:
        """
        시뮬레이션 결과를 파일로 저장합니다.
        
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
        
        # 평균 주차 시간 계산
        park_events = df[df.event.isin(["park_success", "depart"])].copy()
        park_events = park_events.sort_values(["id", "time"])
        total_parking_time = 0
        parking_count = 0
        
        # 각 차량별로 주차 시작과 출차 이벤트를 매칭
        for vehicle_id in park_events.id.unique():
            v_events = park_events[park_events.id == vehicle_id].sort_values("time")
            park_starts = v_events[v_events.event == "park_success"]
            departs = v_events[v_events.event == "depart"]
            
            print(f"\n차량 {vehicle_id}:")
            print(f"주차 성공 이벤트: {len(park_starts)}개")
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
        
        # distance_log.csv에서 평균 이동 거리 계산
        avg_normal_dist = avg_ev_dist = "N/A"
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
                avg_normal_dist = f"{sum(normal_dists)/len(normal_dists):.1f}"
            if ev_dists:
                avg_ev_dist = f"{sum(ev_dists)/len(ev_dists):.1f}"
        except Exception:
            pass
        
        # 만차 시간 계산
        full_capacity_time = 0
        last_time = None
        last_occupied = 0  # 초기값을 0으로 설정
        
        # 시간대별 주차 상태 분석
        time_slots = []
        for _, row in df.iterrows():
            if row['event'] in ['park_success', 'depart']:
                time_slots.append((row['time'], row['event']))
        
        time_slots.sort(key=lambda x: x[0])
        
        # 각 시간대별로 주차 상태 추적
        full_capacity_periods = []
        daily_full_capacity = {}  # 일자별 만차 시간 저장
        
        for time, event in time_slots:
            if event == 'park_success':
                current_occupied = last_occupied + 1
            elif event == 'depart':
                current_occupied = last_occupied - 1
            
            if last_time is not None:
                # 이전 시간대가 만차였다면 그 시간만큼 더함
                if last_occupied >= stats['total_parking_spots']:
                    period = time - last_time
                    full_capacity_time += period
                    full_capacity_periods.append((last_time, time, period))
                    
                    # 일자별 만차 시간 계산
                    start_day = int(last_time / (24 * 3600))
                    end_day = int(time / (24 * 3600))
                    
                    if start_day == end_day:
                        # 같은 날짜 내의 만차 시간
                        if start_day not in daily_full_capacity:
                            daily_full_capacity[start_day] = 0
                        daily_full_capacity[start_day] += period
                    else:
                        # 날짜가 바뀌는 경우
                        # 시작 날짜의 남은 시간
                        if start_day not in daily_full_capacity:
                            daily_full_capacity[start_day] = 0
                        daily_full_capacity[start_day] += (24 * 3600) - (last_time % (24 * 3600))
                        
                        # 중간 날짜들은 하루 전체
                        for day in range(start_day + 1, end_day):
                            if day not in daily_full_capacity:
                                daily_full_capacity[day] = 0
                            daily_full_capacity[day] += 24 * 3600
                        
                        # 마지막 날짜의 시간
                        if end_day not in daily_full_capacity:
                            daily_full_capacity[end_day] = 0
                        daily_full_capacity[end_day] += time % (24 * 3600)
            
            last_time = time
            last_occupied = current_occupied
        
        # 마지막 시간까지 만차였다면 그 시간도 더함
        if last_occupied >= stats['total_parking_spots']:
            period = self.env.now - last_time
            full_capacity_time += period
            full_capacity_periods.append((last_time, self.env.now, period))
            
            # 마지막 만차 기간의 일자별 계산
            start_day = int(last_time / (24 * 3600))
            end_day = int(self.env.now / (24 * 3600))
            
            if start_day == end_day:
                if start_day not in daily_full_capacity:
                    daily_full_capacity[start_day] = 0
                daily_full_capacity[start_day] += period
            else:
                # 시작 날짜의 남은 시간
                if start_day not in daily_full_capacity:
                    daily_full_capacity[start_day] = 0
                daily_full_capacity[start_day] += (24 * 3600) - (last_time % (24 * 3600))
                
                # 중간 날짜들은 하루 전체
                for day in range(start_day + 1, end_day):
                    if day not in daily_full_capacity:
                        daily_full_capacity[day] = 0
                    daily_full_capacity[day] += 24 * 3600
                
                # 마지막 날짜의 시간
                if end_day not in daily_full_capacity:
                    daily_full_capacity[end_day] = 0
                daily_full_capacity[end_day] += self.env.now % (24 * 3600)
        
        # 만차 시간을 분 단위로 변환
        full_capacity_minutes = full_capacity_time / 60
        
        print("\n=== 일자별 만차 시간 ===")
        for day in sorted(daily_full_capacity.keys()):
            minutes = daily_full_capacity[day] / 60
            print(f"Day {day + 1}: {minutes:.1f}분")
        print(f"전체 만차 시간: {full_capacity_minutes:.1f}분")
        
        if full_capacity_periods:
            print("\n만차 기간 상세:")
            for start, end, duration in full_capacity_periods:
                print(f"- 시작: {start/3600:.2f}시간, 종료: {end/3600:.2f}시간, 지속: {duration/60:.2f}분")
        
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
  - 충전 성공: {charge_success}회 (성공률: {(charge_success/charge_try*100 if charge_try > 0 else 0):.1f}%)
  - 충전 실패: {charge_fail}회 (실패율: {(charge_fail/charge_try*100 if charge_try > 0 else 0):.1f}%)

[지표]
- 충전소 개수: {stats['total_charger_spots']}개
- 충전 실패율: {(charge_fail/charge_try*100 if charge_try > 0 else 0):.1f}%
- 평균 distance (일반 차량): {avg_normal_dist}칸
- 평균 distance (EV): {avg_ev_dist}칸
- 일반차 만차 시간: {full_capacity_minutes:.1f}분
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
        park_events = df[df.event.isin(["park_success", "depart"])].copy()
        park_events = park_events.sort_values(["id", "time"])
        total_parking_time = 0
        parking_count = 0
        
        # 각 차량별로 주차 시작과 출차 이벤트를 매칭
        for vehicle_id in park_events.id.unique():
            v_events = park_events[park_events.id == vehicle_id].sort_values("time")
            park_starts = v_events[v_events.event == "park_success"]
            departs = v_events[v_events.event == "depart"]
            
            print(f"\n차량 {vehicle_id}:")
            print(f"주차 성공 이벤트: {len(park_starts)}개")
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
        
        # distance_log.csv에서 평균 이동 거리 계산
        avg_normal_dist = avg_ev_dist = "N/A"
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
                avg_normal_dist = f"{sum(normal_dists)/len(normal_dists):.1f}"
            if ev_dists:
                avg_ev_dist = f"{sum(ev_dists)/len(ev_dists):.1f}"
        except Exception:
            pass
        
        # 만차 시간 계산
        full_capacity_time = 0
        last_time = None
        last_occupied = 0  # 초기값을 0으로 설정
        
        # 시간대별 주차 상태 분석
        time_slots = []
        for _, row in df.iterrows():
            if row['event'] in ['park_success', 'depart']:
                time_slots.append((row['time'], row['event']))
        
        time_slots.sort(key=lambda x: x[0])
        
        # 각 시간대별로 주차 상태 추적
        full_capacity_periods = []
        daily_full_capacity = {}  # 일자별 만차 시간 저장
        
        for time, event in time_slots:
            if event == 'park_success':
                current_occupied = last_occupied + 1
            elif event == 'depart':
                current_occupied = last_occupied - 1
            
            if last_time is not None:
                # 이전 시간대가 만차였다면 그 시간만큼 더함
                if last_occupied >= stats['total_parking_spots']:
                    period = time - last_time
                    full_capacity_time += period
                    full_capacity_periods.append((last_time, time, period))
                    
                    # 일자별 만차 시간 계산
                    start_day = int(last_time / (24 * 3600))
                    end_day = int(time / (24 * 3600))
                    
                    if start_day == end_day:
                        # 같은 날짜 내의 만차 시간
                        if start_day not in daily_full_capacity:
                            daily_full_capacity[start_day] = 0
                        daily_full_capacity[start_day] += period
                    else:
                        # 날짜가 바뀌는 경우
                        # 시작 날짜의 남은 시간
                        if start_day not in daily_full_capacity:
                            daily_full_capacity[start_day] = 0
                        daily_full_capacity[start_day] += (24 * 3600) - (last_time % (24 * 3600))
                        
                        # 중간 날짜들은 하루 전체
                        for day in range(start_day + 1, end_day):
                            if day not in daily_full_capacity:
                                daily_full_capacity[day] = 0
                            daily_full_capacity[day] += 24 * 3600
                        
                        # 마지막 날짜의 시간
                        if end_day not in daily_full_capacity:
                            daily_full_capacity[end_day] = 0
                        daily_full_capacity[end_day] += time % (24 * 3600)
            
            last_time = time
            last_occupied = current_occupied
        
        # 마지막 시간까지 만차였다면 그 시간도 더함
        if last_occupied >= stats['total_parking_spots']:
            period = self.env.now - last_time
            full_capacity_time += period
            full_capacity_periods.append((last_time, self.env.now, period))
            
            # 마지막 만차 기간의 일자별 계산
            start_day = int(last_time / (24 * 3600))
            end_day = int(self.env.now / (24 * 3600))
            
            if start_day == end_day:
                if start_day not in daily_full_capacity:
                    daily_full_capacity[start_day] = 0
                daily_full_capacity[start_day] += period
            else:
                # 시작 날짜의 남은 시간
                if start_day not in daily_full_capacity:
                    daily_full_capacity[start_day] = 0
                daily_full_capacity[start_day] += (24 * 3600) - (last_time % (24 * 3600))
                
                # 중간 날짜들은 하루 전체
                for day in range(start_day + 1, end_day):
                    if day not in daily_full_capacity:
                        daily_full_capacity[day] = 0
                    daily_full_capacity[day] += 24 * 3600
                
                # 마지막 날짜의 시간
                if end_day not in daily_full_capacity:
                    daily_full_capacity[end_day] = 0
                daily_full_capacity[end_day] += self.env.now % (24 * 3600)
        
        # 만차 시간을 분 단위로 변환
        full_capacity_minutes = full_capacity_time / 60
        
        print("\n=== 일자별 만차 시간 ===")
        for day in sorted(daily_full_capacity.keys()):
            minutes = daily_full_capacity[day] / 60
            print(f"Day {day + 1}: {minutes:.1f}분")
        print(f"전체 만차 시간: {full_capacity_minutes:.1f}분")
        
        if full_capacity_periods:
            print("\n만차 기간 상세:")
            for start, end, duration in full_capacity_periods:
                print(f"- 시작: {start/3600:.2f}시간, 종료: {end/3600:.2f}시간, 지속: {duration/60:.2f}분")
        
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
  - 충전 성공: {charge_success}회 (성공률: {(charge_success/charge_try*100 if charge_try > 0 else 0):.1f}%)
  - 충전 실패: {charge_fail}회 (실패율: {(charge_fail/charge_try*100 if charge_try > 0 else 0):.1f}%)

[지표]
- 충전소 개수: {stats['total_charger_spots']}개
- 충전 실패율: {(charge_fail/charge_try*100 if charge_try > 0 else 0):.1f}%
- 평균 distance (일반 차량): {avg_normal_dist}칸
- 평균 distance (EV): {avg_ev_dist}칸
- 일반차 만차 시간: {full_capacity_minutes:.1f}분
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