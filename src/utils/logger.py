"""
시뮬레이션 이벤트를 기록하고 분석하는 로깅 시스템입니다.
"""
from typing import List, Dict, Any, Optional
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from datetime import datetime
import platform
import numpy as np
import json
import os
import time
import csv

# 한글 폰트 설정
if platform.system() == 'Windows':
    plt.rcParams['font.family'] = 'Malgun Gothic'  # 윈도우 한글 폰트
elif platform.system() == 'Darwin':  # macOS
    plt.rcParams['font.family'] = 'AppleGothic'    # 맥OS 한글 폰트
else:  # Linux
    plt.rcParams['font.family'] = 'NanumGothic'    # 리눅스 한글 폰트

mpl.rcParams['axes.unicode_minus'] = False   # 마이너스 기호 깨짐 방지

# 로그 엔트리 타입 정의
LogEntry = Dict[str, Any]
ChargeLogEntry = Dict[str, Any]


class SimulationLogger:
    """시뮬레이션 이벤트를 기록하고 분석하는 클래스"""
    
    def __init__(self, log_file: str, stats_file: str):
        """
        로거를 초기화합니다.
        
        Args:
            log_file: 로그 파일 경로
            stats_file: 통계 파일 경로
        """
        self.log_file = log_file
        self.stats_file = stats_file
        
        # 로그 리스트 초기화
        self.log: List[LogEntry] = []
        self.charge_log: List[ChargeLogEntry] = []
        
        # 로그 파일 초기화
        with open(self.log_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['time', 'vehicle_id', 'vehicle_type', 'event', 'floor', 'row', 'col', 'battery', 'building', 'parking_duration'])
        
        # 통계 초기화
        self.stats = {
            "total_entries": 0,
            "successful_parks": 0,
            "failed_parks": 0,
            "total_charges": 0,
            "daily_park_fails": {}  # 일자별 park_fail 통계 추가
        }
        
        self.arrivals_graph_path = 'arrivals_by_hour.png'
        self.charge_graph_path = 'charging_patterns.png'
    
    def set_graph_paths(self, arrivals_path: str, charge_path: str = None) -> None:
        """
        그래프 저장 경로를 설정합니다.
        
        Args:
            arrivals_path: 시간대별 차량 도착 그래프 저장 경로
            charge_path: 충전 패턴 그래프 저장 경로
        """
        self.arrivals_graph_path = arrivals_path
        if charge_path:
            self.charge_graph_path = charge_path
    
    def add_event(self, vehicle_id: int, vehicle_type: str, event: str, 
                 time: float, pos: Optional[tuple] = None, battery: Optional[float] = None,
                 building: Optional[str] = None, floor: Optional[str] = None) -> None:
        """
        시뮬레이션 이벤트를 로그에 추가합니다.
        
        Args:
            vehicle_id: 차량 ID
            vehicle_type: 차량 타입 ("normal" 또는 "ev")
            event: 이벤트 유형 (arrive, park_start, charge_start, charge_complete, depart, move)
            time: 이벤트 발생 시간 (시뮬레이션 시간, 초 단위)
            pos: 이벤트 발생 위치 (r, c), 선택적
            battery: 전기차의 배터리 잔량 (0-100%)
            building: 차량이 속한 건물 동
            floor: 주차한 층
        """
        # charge_update 이벤트는 로깅하지 않음
        if event == "charge_update":
            return
            
        entry: LogEntry = {
            "id": vehicle_id,
            "type": vehicle_type,
            "event": event,
            "time": time,
            "pos_r": pos[0] if pos else None,
            "pos_c": pos[1] if pos else None,
            "battery": battery,
            "building": building,
            "floor": floor
        }
        self.log.append(entry)
        
        # 충전 관련 이벤트는 충전 로그에도 기록
        if event in ["charge_start", "charge_complete"] and vehicle_type == "ev":
            charge_entry: ChargeLogEntry = {
                "id": vehicle_id,
                "event": event,
                "time": time,
                "battery": battery,
                "building": building,
                "floor": floor,
                "hour": time / 3600  # 시간 단위로 변환
            }
            self.charge_log.append(charge_entry)
            
            # 충전 상태 변화 출력
            if event == "charge_start":
                print(f"[충전 시작] 차량 {vehicle_id}: 배터리 {battery:.1f}%")
            elif event == "charge_complete":
                print(f"[충전 완료] 차량 {vehicle_id}: 배터리 {battery:.1f}%")
    
    def get_dataframe(self) -> pd.DataFrame:
        """로그를 판다스 DataFrame으로 변환해 반환합니다."""
        return pd.DataFrame(self.log)
    
    def get_charge_dataframe(self) -> pd.DataFrame:
        """충전 관련 이벤트만 포함하는 DataFrame 반환"""
        df = self.get_dataframe()
        return df[df.event.isin(["charge_start", "charge_end", "charge_fail"])]
    
    def save_to_csv(self, filename: Optional[str] = None) -> str:
        """로그 데이터를 CSV 파일로 저장"""
        if filename is None:
            filename = f"simulation_log_{int(time.time())}.csv"
        df = self.get_dataframe()
        df.to_csv(filename, index=False)
        return filename
    
    def print_summary(self) -> None:
        """시뮬레이션 결과 요약을 출력합니다."""
        df = self.get_dataframe()
        
        print("=== 시뮬레이션 요약 ===")
        print(f"총 이벤트 수: {len(df)}")
        print("\n이벤트 유형별 분포:")
        print(df.groupby("event").size())
        
        # 전기차 충전 관련 요약
        ev_df = df[df.type == "ev"]
        charge_events = ev_df[ev_df.event == "charge_start"].shape[0]
        
        # 충전소가 사용된 총 시간 계산
        if "charge_start" in df.event.values and "charge_end" in df.event.values:
            charge_df = df[df.event.isin(["charge_start", "charge_end"])]
            charge_df = charge_df.sort_values(["id", "time"])
            
            total_charge_time = 0
            for vehicle_id in charge_df.id.unique():
                v_events = charge_df[charge_df.id == vehicle_id]
                
                for i in range(0, len(v_events), 2):
                    if i+1 < len(v_events):
                        start_time = v_events.iloc[i].time
                        end_time = v_events.iloc[i+1].time
                        total_charge_time += (end_time - start_time)
            
            max_time = df.time.max()
            print(f"\n충전소 평균 활용률: {total_charge_time / (max_time * 4) * 100:.2f}%")
        
        print(f"총 충전 시도 수: {charge_events}")
        
        # 차량 유형 분포
        print("\n차량 유형 분포:")
        vehicle_counts = df.drop_duplicates("id").groupby("type").size()
        print(vehicle_counts)
        
        # 충전 로그 요약
        charge_df = self.get_charge_dataframe()
        if not charge_df.empty:
            print("\n충전 로그 요약:")
            print(f"총 충전 이벤트 수: {len(charge_df)}")
            charge_by_event = charge_df.groupby("event").size()
            print(charge_by_event)
            
            # 배터리 증가 통계
            charge_starts = charge_df[charge_df.event == "charge_start"]
            charge_ends = charge_df[charge_df.event == "charge_end"]
            
            if not charge_starts.empty and not charge_ends.empty:
                print("\n배터리 충전 통계:")
                battery_gains = []
                
                for vehicle_id in charge_starts.id.unique():
                    start = charge_starts[charge_starts.id == vehicle_id]
                    end = charge_ends[charge_ends.id == vehicle_id]
                    
                    if not start.empty and not end.empty:
                        start_battery = start.iloc[0].battery
                        end_battery = end.iloc[0].battery
                        gain = end_battery - start_battery
                        battery_gains.append(gain)
                
                if battery_gains:
                    avg_gain = sum(battery_gains) / len(battery_gains)
                    print(f"평균 배터리 충전량: {avg_gain:.2f}%")
    
    def generate_plots(self, results_dir: str) -> None:
        """시뮬레이션 결과를 그래프로 시각화"""
        df = self.get_dataframe()
        
        # 1. 시간대별 주차장 점유율
        self._plot_parking_occupancy(df, results_dir)
        
        # 2. 시간대별 충전기 사용률
        self._plot_charger_usage(df, results_dir)
        
        # 3. 시간대별 충전 시도/성공/실패
        self._plot_charging_attempts(df, results_dir)
        
        # 4. 시간대별 주차 시도/성공/실패
        self._plot_parking_attempts(df, results_dir)
    
    def _plot_parking_occupancy(self, df: pd.DataFrame, results_dir: str) -> None:
        """시간대별 주차장 점유율 그래프"""
        # 주차 시작/종료 이벤트만 추출
        park_events = df[df.event.isin(["park_start", "park_end"])].copy()
        
        # 시간을 1시간 단위로 그룹화
        park_events["hour"] = park_events["time"] // 3600
        
        # 각 시간대별 주차 시작/종료 수 계산
        hourly_stats = park_events.groupby(["hour", "event"]).size().unstack(fill_value=0)
        
        # 누적 주차 수 계산
        if "park_start" in hourly_stats.columns and "park_end" in hourly_stats.columns:
            hourly_stats["current"] = hourly_stats["park_start"].cumsum() - hourly_stats["park_end"].cumsum()
        else:
            hourly_stats["current"] = 0
            
        # 그래프 생성
        plt.figure(figsize=(12, 6))
        plt.plot(hourly_stats.index, hourly_stats["current"], marker="o")
        plt.title("시간대별 주차장 점유율")
        plt.xlabel("시간")
        plt.ylabel("주차 중인 차량 수")
        plt.grid(True)
        
        # 그래프 저장
        plt.savefig(os.path.join(results_dir, "parking_occupancy.png"))
        plt.close()
        
    def _plot_charger_usage(self, df: pd.DataFrame, results_dir: str) -> None:
        """시간대별 충전기 사용률 그래프"""
        # 충전 시작/종료 이벤트만 추출
        charge_events = df[df.event.isin(["charge_start", "charge_end"])].copy()
        
        # 시간을 1시간 단위로 그룹화
        charge_events["hour"] = charge_events["time"] // 3600
        
        # 각 시간대별 충전 시작/종료 수 계산
        hourly_stats = charge_events.groupby(["hour", "event"]).size().unstack(fill_value=0)
        
        # 누적 충전 수 계산
        if "charge_start" in hourly_stats.columns and "charge_end" in hourly_stats.columns:
            hourly_stats["current"] = hourly_stats["charge_start"].cumsum() - hourly_stats["charge_end"].cumsum()
        else:
            hourly_stats["current"] = 0
            
        # 그래프 생성
        plt.figure(figsize=(12, 6))
        plt.plot(hourly_stats.index, hourly_stats["current"], marker="o")
        plt.title("시간대별 충전기 사용률")
        plt.xlabel("시간")
        plt.ylabel("충전 중인 차량 수")
        plt.grid(True)
        
        # 그래프 저장
        plt.savefig(os.path.join(results_dir, "charger_usage.png"))
        plt.close()
        
    def _plot_charging_attempts(self, df: pd.DataFrame, results_dir: str) -> None:
        """시간대별 충전 시도/성공/실패 그래프"""
        # 충전 관련 이벤트만 추출
        charge_events = df[df.event.isin(["charge_start", "charge_fail"])].copy()
        
        # 시간을 1시간 단위로 그룹화
        charge_events["hour"] = charge_events["time"] // 3600
        
        # 각 시간대별 이벤트 수 계산
        hourly_stats = charge_events.groupby(["hour", "event"]).size().unstack(fill_value=0)
        
        # 그래프 생성
        plt.figure(figsize=(12, 6))
        if "charge_start" in hourly_stats.columns:
            plt.plot(hourly_stats.index, hourly_stats["charge_start"], marker="o", label="충전 시도")
        if "charge_fail" in hourly_stats.columns:
            plt.plot(hourly_stats.index, hourly_stats["charge_fail"], marker="x", label="충전 실패")
        plt.title("시간대별 충전 시도/실패")
        plt.xlabel("시간")
        plt.ylabel("횟수")
        plt.legend()
        plt.grid(True)
        
        # 그래프 저장
        plt.savefig(os.path.join(results_dir, "charging_attempts.png"))
        plt.close()
        
    def _plot_parking_attempts(self, df: pd.DataFrame, results_dir: str) -> None:
        """시간대별 주차 시도/성공/실패 그래프"""
        # 주차 관련 이벤트만 추출
        park_events = df[df.event.isin(["park_start", "park_fail"])].copy()
        
        # 시간을 1시간 단위로 그룹화
        park_events["hour"] = park_events["time"] // 3600
        
        # 각 시간대별 이벤트 수 계산
        hourly_stats = park_events.groupby(["hour", "event"]).size().unstack(fill_value=0)
        
        # 그래프 생성
        plt.figure(figsize=(12, 6))
        if "park_start" in hourly_stats.columns:
            plt.plot(hourly_stats.index, hourly_stats["park_start"], marker="o", label="주차 시도")
        if "park_fail" in hourly_stats.columns:
            plt.plot(hourly_stats.index, hourly_stats["park_fail"], marker="x", label="주차 실패")
        plt.title("시간대별 주차 시도/실패")
        plt.xlabel("시간")
        plt.ylabel("횟수")
        plt.legend()
        plt.grid(True)
        
        # 그래프 저장
        plt.savefig(os.path.join(results_dir, "parking_attempts.png"))
        plt.close()

    def calculate_charger_cost(self, charger_count: int, charger_price: int = 800000, maintenance_per_year: int = 100000, lifetime_years: int = 7) -> int:
        """
        충전소 설치 및 유지 총비용 계산
        Args:
            charger_count: 충전기 개수
            charger_price: 충전기 1대당 설치비(원)
            maintenance_per_year: 연간 유지비(원)
            lifetime_years: 충전기 수명(년)
        Returns:
            총비용(원)
        """
        설치비 = charger_count * charger_price
        유지비 = charger_count * maintenance_per_year * lifetime_years
        총비용 = 설치비 + 유지비
        return 총비용

    def calculate_charger_idle_rate(self, sim_time: float, charger_count: int) -> float:
        """
        충전소 공실률 계산 (실제 시간별 점유 충전소 개수 기반)
        Args:
            sim_time: 시뮬레이션 전체 시간(초)
            charger_count: 충전기 개수
        Returns:
            공실률(0~1), 충전소가 0개일 경우 0.0 반환
        """
        # 충전소가 0개인 경우
        if charger_count == 0:
            return 0.0
            
        df = self.get_dataframe()
        # 충전 시작/종료 이벤트만 추출, 시간순 정렬
        charge_events = df[df.event.isin(["charge_start", "charge_end"])].sort_values("time")
        
        # (시간, +1/-1) 리스트 생성
        timeline = []
        for _, row in charge_events.iterrows():
            if row["event"] == "charge_start":
                timeline.append((row["time"], +1))
            elif row["event"] == "charge_end":
                timeline.append((row["time"], -1))
        timeline.sort()
        
        # 시뮬레이션 시작~끝까지, 각 구간별 점유 충전소 개수 누적
        last_time = 0.0
        current_occupied = 0
        total_occupied_time = 0.0
        for time, delta in timeline:
            duration = time - last_time
            total_occupied_time += duration * current_occupied
            current_occupied += delta
            last_time = time
        # 마지막 구간(마지막 이벤트~시뮬레이션 종료)도 반영
        if last_time < sim_time:
            total_occupied_time += (sim_time - last_time) * current_occupied
        전체_충전기_가동가능시간 = sim_time * charger_count
        공실률 = 1 - (total_occupied_time / 전체_충전기_가동가능시간)
        return max(0.0, min(1.0, 공실률))

    def calculate_charge_fail_rate(self) -> float:
        """
        전기차 충전 실패율 계산 (충전 시도 대비 실패 비율)
        Returns:
            실패율(0~1)
        """
        df = self.get_dataframe()
        ev_df = df[df.type == "ev"]
        # 충전 시도: charge_start 이벤트 수
        total_attempts = ev_df[ev_df.event == "charge_start"].shape[0]
        # 충전 실패: charge_fail 이벤트 수 (이벤트명이 실제로 기록되는지 확인 필요)
        fail_count = ev_df[ev_df.event == "charge_fail"].shape[0]
        if total_attempts == 0:
            return 0.0
        return fail_count / total_attempts

    def calculate_parking_fail_rate(self) -> float:
        """
        주차 실패율 계산 (주차 시도 대비 실패 비율, EV/일반차 공통)
        Returns:
            실패율(0~1)
        """
        df = self.get_dataframe()
        # 주차 시도: park_start 이벤트 수
        total_attempts = df[df.event == "park_start"].shape[0]
        # 주차 실패: park_fail 이벤트 수 (이벤트명이 실제로 기록되는지 확인 필요)
        fail_count = df[df.event == "park_fail"].shape[0]
        if total_attempts == 0:
            return 0.0
        return fail_count / total_attempts

    def log_event(self, time: float, vehicle_id: str, event: str, vehicle_type: str = None, 
                 floor: str = None, pos: tuple = None, battery: float = None, 
                 building: str = None, parking_duration: float = None):
        """이벤트 로깅"""
        # 이벤트 저장
        event_data = {
            'time': time,
            'vehicle_id': vehicle_id,
            'vehicle_type': vehicle_type,
            'event': event,
            'floor': floor,
            'row': pos[0] if pos else None,
            'col': pos[1] if pos else None,
            'battery': battery,
            'building': building,
            'parking_duration': parking_duration
        }
        self.log.append(event_data)
        
        # CSV 파일에 기록
        with open(self.log_file, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                time, vehicle_id, vehicle_type, event, floor,
                pos[0] if pos else '', pos[1] if pos else '',
                battery, building, parking_duration
            ])
        
        # park_fail 이벤트인 경우 일자별 통계 업데이트
        if event == "park_fail":
            day = int(time // 86400)  # 일자 계산 (86400초 = 24시간)
            if day not in self.stats["daily_park_fails"]:
                self.stats["daily_park_fails"][day] = 0
            self.stats["daily_park_fails"][day] += 1

    def update_stats(self, event: str, duration: float = 0) -> None:
        """
        통계 정보를 업데이트합니다.
        
        Args:
            event: 이벤트 유형
            duration: 소요 시간
        """
        if event == "park_success":
            self.stats["successful_parks"] += 1
            self.stats["avg_parking_time"] = (
                (self.stats["avg_parking_time"] * (self.stats["successful_parks"] - 1) + duration)
                / self.stats["successful_parks"]
            )
        elif event == "charge_comp":
            self.stats["total_charges"] += 1
            self.stats["avg_charging_time"] = (
                (self.stats["avg_charging_time"] * (self.stats["total_charges"] - 1) + duration)
                / self.stats["total_charges"]
            )
    
    def save_stats(self) -> None:
        """통계 정보를 JSON 파일로 저장"""
        with open(self.stats_file, 'w', encoding='utf-8') as f:
            json.dump(self.stats, f, ensure_ascii=False, indent=2)
            
        # 일자별 park_fail 통계 출력
        print("\n=== 일자별 Park Fail 통계 ===")
        for day, count in sorted(self.stats["daily_park_fails"].items()):
            print(f"Day {day}: {count}회")
        print("==========================\n") 