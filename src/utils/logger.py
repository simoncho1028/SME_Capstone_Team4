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
        with open(self.log_file, 'w', encoding='utf-8') as f:
            f.write("time,id,event,floor,pos_r,pos_c,battery,parking_duration\n")
        
        # 통계 초기화
        self.stats = {
            "total_vehicles": 0,
            "total_parked": 0,
            "total_charged": 0,
            "avg_parking_time": 0,
            "avg_charging_time": 0
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
        """충전 로그를 판다스 DataFrame으로 변환해 반환합니다."""
        return pd.DataFrame(self.charge_log)
    
    def save_to_csv(self, filename: Optional[str] = None) -> str:
        """
        로그를 CSV 파일로 저장합니다.
        
        Args:
            filename: 저장할 파일 이름 (없으면 타임스탬프로 자동 생성)
            
        Returns:
            저장된 파일 경로
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"simulation_log_{timestamp}.csv"
        
        df = self.get_dataframe()
        df.to_csv(filename, index=False)
        
        # 충전 로그 저장
        charge_df = self.get_charge_dataframe()
        if not charge_df.empty:
            charge_filename = filename.replace("simulation_log", "charge_log")
            charge_df.to_csv(charge_filename, index=False)
            print(f"충전 로그가 {charge_filename}에 저장되었습니다.")
        
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
    
    def generate_plots(self) -> None:
        """시뮬레이션 결과를 비즈니스 관점에서 시각화하는 그래프를 생성합니다."""
        df = self.get_dataframe()
        
        if df.empty:
            print("[WARNING] 데이터가 없어 그래프를 생성할 수 없습니다.")
            return
        
        # 시간 단위로 집계
        df['hour'] = df.time / 3600
        
        # 1. 주차장 운영 현황 종합 분석 (4개 차트)
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('주차장 운영 현황 종합 분석', fontsize=16, fontweight='bold')
        
        # 1-1. 시간대별 차량 도착 패턴 (개선)
        arrivals = df[df.event == "arrive"].copy()
        arrivals['hour_bin'] = arrivals.hour.apply(lambda x: int(x))
        
        # 차량 유형별로 구분
        ev_arrivals = arrivals[arrivals.type == "ev"].groupby('hour_bin').size()
        normal_arrivals = arrivals[arrivals.type == "normal"].groupby('hour_bin').size()
        
        # 모든 시간대 확보 (0-23시)
        all_hours = range(24)
        ev_counts = [ev_arrivals.get(h, 0) for h in all_hours]
        normal_counts = [normal_arrivals.get(h, 0) for h in all_hours]
        
        ax1.bar(all_hours, normal_counts, label='일반 차량', color='skyblue', alpha=0.8)
        ax1.bar(all_hours, ev_counts, bottom=normal_counts, label='전기차', color='orange', alpha=0.8)
        ax1.set_title('시간대별 차량 도착 패턴', fontweight='bold')
        ax1.set_xlabel('시간(시)')
        ax1.set_ylabel('도착 차량 수')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_xticks(range(0, 24, 2))
        
        # 1-2. 충전소 이용률 분석
        charge_df = self.get_charge_dataframe()
        if not charge_df.empty:
            charge_usage = charge_df[charge_df.event == "charge_start"].copy()
            charge_usage['hour_bin'] = charge_usage.hour.apply(lambda x: int(x))
            charge_hourly = charge_usage.groupby('hour_bin').size()
            
            hourly_usage = [charge_hourly.get(h, 0) for h in all_hours]
            ax2.plot(all_hours, hourly_usage, marker='o', linewidth=2, markersize=6, color='green')
            ax2.fill_between(all_hours, hourly_usage, alpha=0.3, color='green')
            ax2.set_title('시간대별 충전소 이용률', fontweight='bold')
            ax2.set_xlabel('시간(시)')
            ax2.set_ylabel('충전 시작 횟수')
            ax2.grid(True, alpha=0.3)
            ax2.set_xticks(range(0, 24, 2))
        else:
            ax2.text(0.5, 0.5, '충전 데이터 없음', ha='center', va='center', transform=ax2.transAxes)
            ax2.set_title('시간대별 충전소 이용률', fontweight='bold')
        
        # 1-3. 주차장 점유율 시간대별 분석
        parking_occupancy = []
        for hour in all_hours:
            start_time = hour * 3600
            parked_at_hour = 0
            for vehicle_id in df.id.unique():
                vehicle_events = df[df.id == vehicle_id].sort_values('time')
                park_events = vehicle_events[vehicle_events.event.isin(['park_success', 'depart'])]
                
                if len(park_events) >= 1:
                    park_time = park_events.iloc[0].time if park_events.iloc[0].event == 'park_success' else None
                    depart_time = park_events[park_events.event == 'depart'].iloc[0].time if len(park_events[park_events.event == 'depart']) > 0 else float('inf')
                    
                    if park_time and park_time <= start_time and depart_time > start_time:
                        parked_at_hour += 1
            
            parking_occupancy.append(parked_at_hour)
        
        ax3.plot(all_hours, parking_occupancy, marker='s', linewidth=2, markersize=6, color='red')
        ax3.fill_between(all_hours, parking_occupancy, alpha=0.3, color='red')
        ax3.set_title('시간대별 주차장 점유율', fontweight='bold')
        ax3.set_xlabel('시간(시)')
        ax3.set_ylabel('점유 차량 수')
        ax3.grid(True, alpha=0.3)
        ax3.set_xticks(range(0, 24, 2))
        
        # 1-4. 주차 성공률 및 효율성 지표
        total_arrivals = len(df[df.event == "arrive"])
        successful_parks = len(df[df.event == "park_success"])
        failed_parks = len(df[df.event == "park_fail"])
        
        if total_arrivals > 0:
            success_rate = (successful_parks / total_arrivals) * 100
            fail_rate = (failed_parks / total_arrivals) * 100
            
            metrics = ['주차 성공률', '주차 실패율']
            values = [success_rate, fail_rate]
            colors = ['#2ECC71', '#E74C3C']
            
            bars = ax4.bar(metrics, values, color=colors, alpha=0.8)
            ax4.set_title('주차장 운영 효율성 지표', fontweight='bold')
            ax4.set_ylabel('비율 (%)')
            ax4.set_ylim(0, 100)
            
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax4.text(bar.get_x() + bar.get_width()/2., height + 1, f'{value:.1f}%', 
                        ha='center', va='bottom', fontweight='bold')
        else:
            ax4.text(0.5, 0.5, '주차 데이터 없음', ha='center', va='center', transform=ax4.transAxes)
            ax4.set_title('주차장 운영 효율성 지표', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.arrivals_graph_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. 비즈니스 성과 분석 (4개 차트)
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('주차장 비즈니스 성과 분석', fontsize=16, fontweight='bold')
        
        # 2-1. 층별 이용률 분포
        floor_usage = df[df.floor.notna()].groupby('floor').size()
        if not floor_usage.empty:
            colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
            wedges, texts, autotexts = ax1.pie(floor_usage.values, labels=floor_usage.index, 
                                               autopct='%1.1f%%', colors=colors, startangle=90)
            ax1.set_title('층별 이용률 분포', fontweight='bold')
            
            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_fontweight('bold')
        else:
            ax1.text(0.5, 0.5, '층별 데이터 없음', ha='center', va='center', transform=ax1.transAxes)
            ax1.set_title('층별 이용률 분포', fontweight='bold')
        
        # 2-2. 차량 유형별 비율
        vehicle_types = df.drop_duplicates('id').groupby('type').size()
        if len(vehicle_types) > 1:
            colors = ['#FFD93D', '#6BCF7F']
            ax2.bar(vehicle_types.index, vehicle_types.values, color=colors, alpha=0.8)
            ax2.set_title('차량 유형별 비율', fontweight='bold')
            ax2.set_ylabel('차량 수')
            
            for i, v in enumerate(vehicle_types.values):
                ax2.text(i, v + max(vehicle_types.values) * 0.01, str(v), 
                        ha='center', va='bottom', fontweight='bold')
        else:
            ax2.text(0.5, 0.5, '차량 유형 데이터 부족', ha='center', va='center', transform=ax2.transAxes)
            ax2.set_title('차량 유형별 비율', fontweight='bold')
        
        # 2-3. 시간대별 충전소 예상 수익
        if not charge_df.empty:
            charge_revenue_hourly = charge_df[charge_df.event == "charge_start"].copy()
            charge_revenue_hourly['hour_bin'] = charge_revenue_hourly.hour.apply(lambda x: int(x))
            hourly_revenue = charge_revenue_hourly.groupby('hour_bin').size() * 5000  # 5000원/회
            
            revenue_by_hour = [hourly_revenue.get(h, 0) for h in all_hours]
            ax3.bar(all_hours, revenue_by_hour, color='gold', alpha=0.8)
            ax3.set_title('시간대별 충전소 예상 수익', fontweight='bold')
            ax3.set_xlabel('시간(시)')
            ax3.set_ylabel('예상 수익 (원)')
            ax3.grid(True, alpha=0.3)
            ax3.set_xticks(range(0, 24, 2))
            ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x/1000)}K'))
        else:
            ax3.text(0.5, 0.5, '충전 수익 데이터 없음', ha='center', va='center', transform=ax3.transAxes)
            ax3.set_title('시간대별 충전소 예상 수익', fontweight='bold')
        
        # 2-4. 주차 수요 vs 공급 분석
        total_spots = df[df.event == "park_success"].shape[0] + df[df.event == "park_fail"].shape[0]
        success_spots = df[df.event == "park_success"].shape[0]
        
        if total_spots > 0:
            supply_demand = ['공급 가능', '수요 초과']
            values = [success_spots, total_spots - success_spots]
            colors = ['#27AE60', '#E67E22']
            
            wedges, texts, autotexts = ax4.pie(values, labels=supply_demand, autopct='%1.1f%%',
                                               colors=colors, startangle=90)
            ax4.set_title('주차 수요 vs 공급 분석', fontweight='bold')
            
            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_fontweight('bold')
        else:
            ax4.text(0.5, 0.5, '주차 수요/공급 데이터 없음', ha='center', va='center', transform=ax4.transAxes)
            ax4.set_title('주차 수요 vs 공급 분석', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.arrivals_graph_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. 개선된 전기차 충전 패턴 분석
        if not charge_df.empty:
            plt.figure(figsize=(14, 8))
            
            unique_vehicles = charge_df.id.unique()[:10]  # 최대 10대만 표시
            colors = plt.cm.tab10(np.linspace(0, 1, len(unique_vehicles)))
            
            for i, vehicle_id in enumerate(unique_vehicles):
                v_data = charge_df[charge_df.id == vehicle_id].sort_values('time')
                if len(v_data) > 1 and not v_data.battery.isna().all():
                    plt.plot(v_data.time / 60, v_data.battery, marker='o', linewidth=2, 
                            markersize=4, label=f'EV #{vehicle_id}', color=colors[i])
            
            plt.title('전기차 충전 패턴 분석 (상위 10대)', fontsize=14, fontweight='bold')
            plt.xlabel('시간 (분)')
            plt.ylabel('배터리 (%)')
            plt.grid(True, alpha=0.3)
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            
            plt.tight_layout()
            plt.savefig(self.charge_graph_path, dpi=300, bbox_inches='tight')
            plt.close()
        
        print(f"비즈니스 분석 그래프가 생성되었습니다:")
        print(f"  - 주차장 운영 현황: {self.arrivals_graph_path}")
        if not charge_df.empty:
            print(f"  - 충전 패턴 분석: {self.charge_graph_path}")

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

    def log_event(self, time: float, vehicle_id: str, event: str, 
                 floor: str = "", pos: tuple = None, battery: float = None,
                 parking_duration: float = None) -> None:
        """
        시뮬레이션 이벤트를 로그 파일에 기록합니다.
        
        Args:
            time: 이벤트 발생 시간
            vehicle_id: 차량 ID
            event: 이벤트 유형
            floor: 주차 층
            pos: 주차 위치 (row, col)
            battery: 배터리 잔량
            parking_duration: 주차 예정 시간
        """
        pos_r = pos[0] if pos else ""
        pos_c = pos[1] if pos else ""
        
        # 로그 파일에 이벤트 기록
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(f"{time},{vehicle_id},{event},{floor},{pos_r},{pos_c},{battery},{parking_duration}\n")
            
        # 통계 업데이트
        if event == "park_success" and parking_duration is not None:
            self.update_stats(event, parking_duration)
    
    def update_stats(self, event: str, duration: float = 0) -> None:
        """
        통계 정보를 업데이트합니다.
        
        Args:
            event: 이벤트 유형
            duration: 소요 시간
        """
        if event == "park_success":
            self.stats["total_parked"] += 1
            self.stats["avg_parking_time"] = (
                (self.stats["avg_parking_time"] * (self.stats["total_parked"] - 1) + duration)
                / self.stats["total_parked"]
            )
        elif event == "charge_comp":
            self.stats["total_charged"] += 1
            self.stats["avg_charging_time"] = (
                (self.stats["avg_charging_time"] * (self.stats["total_charged"] - 1) + duration)
                / self.stats["total_charged"]
            )
    
    def save_stats(self) -> None:
        """통계 정보를 파일에 저장합니다."""
        with open(self.stats_file, 'w', encoding='utf-8') as f:
            json.dump(self.stats, f, ensure_ascii=False, indent=2) 