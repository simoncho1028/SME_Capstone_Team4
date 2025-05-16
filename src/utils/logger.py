"""
시뮬레이션 이벤트를 기록하고 분석하는 로깅 시스템입니다.
"""
from typing import List, Dict, Any, Optional
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from datetime import datetime

# 한글 폰트 설정
plt.rcParams['font.family'] = 'AppleGothic'  # 맥OS 한글 폰트
mpl.rcParams['axes.unicode_minus'] = False   # 마이너스 기호 깨짐 방지

# 로그 엔트리 타입 정의
LogEntry = Dict[str, Any]
ChargeLogEntry = Dict[str, Any]


class SimulationLogger:
    """시뮬레이션 이벤트를 기록하고 분석하는 클래스"""
    
    def __init__(self):
        """로거 초기화"""
        # 일반 이벤트 로그
        self.log: List[LogEntry] = []
        # 충전 관련 로그 (충전 이벤트만 별도 관리)
        self.charge_log: List[ChargeLogEntry] = []
        
        self.arrivals_graph_path = 'arrivals_by_hour.png'
        self.parking_duration_graph_path = 'parking_duration.png'
        self.charge_graph_path = 'charging_patterns.png'
    
    def set_graph_paths(self, arrivals_path: str, parking_duration_path: str, charge_path: str = None) -> None:
        """
        그래프 저장 경로를 설정합니다.
        
        Args:
            arrivals_path: 시간대별 차량 도착 그래프 저장 경로
            parking_duration_path: 주차 시간 분포 그래프 저장 경로
            charge_path: 충전 패턴 그래프 저장 경로
        """
        self.arrivals_graph_path = arrivals_path
        self.parking_duration_graph_path = parking_duration_path
        if charge_path:
            self.charge_graph_path = charge_path
    
    def add_event(self, vehicle_id: int, vehicle_type: str, event: str, 
                 time: float, pos: tuple, battery: Optional[float] = None) -> None:
        """
        시뮬레이션 이벤트를 로그에 추가합니다.
        
        Args:
            vehicle_id: 차량 ID
            vehicle_type: 차량 타입 ("normal" 또는 "ev")
            event: 이벤트 유형 (arrive, park_start, charge_start, charge_update, charge_end, depart)
            time: 이벤트 발생 시간 (시뮬레이션 시간, 초 단위)
            pos: 이벤트 발생 위치 (r, c)
            battery: 전기차의 배터리 잔량 (0-100%)
        """
        entry: LogEntry = {
            "id": vehicle_id,
            "type": vehicle_type,
            "event": event,
            "time": time,
            "pos_r": pos[0],
            "pos_c": pos[1],
            "battery": battery
        }
        self.log.append(entry)
        
        # 충전 관련 이벤트는 충전 로그에도 기록
        if event in ["charge_start", "charge_update", "charge_end"] and vehicle_type == "ev":
            charge_entry: ChargeLogEntry = {
                "id": vehicle_id,
                "event": event,
                "time": time,
                "battery": battery
            }
            self.charge_log.append(charge_entry)
    
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
        """시뮬레이션 결과를 시각화하는 그래프를 생성합니다."""
        df = self.get_dataframe()
        
        # 시간 단위로 집계
        df['hour'] = df.time / 3600
        
        # 1. 시간대별 차량 도착 그래프
        plt.figure(figsize=(10, 6))
        arrivals = df[df.event == "arrive"].copy()
        arrivals['hour_bin'] = arrivals.hour.apply(lambda x: int(x))
        hour_counts = arrivals.groupby('hour_bin').size()
        
        plt.bar(hour_counts.index, hour_counts.values)
        plt.title('시간대별 차량 도착 수')
        plt.xlabel('시간(시)')
        plt.ylabel('도착 차량 수')
        plt.savefig(self.arrivals_graph_path)
        
        # 2. 차량 유형별 주차 시간 분포
        plt.figure(figsize=(10, 6))
        
        parking_times = []
        for v_id in df.id.unique():
            v_df = df[df.id == v_id]
            arrive = v_df[v_df.event == "arrive"]
            depart = v_df[v_df.event == "depart"]
            
            if not arrive.empty and not depart.empty:
                arrive_time = arrive.iloc[0].time
                depart_time = depart.iloc[0].time
                duration = (depart_time - arrive_time) / 3600  # 시간 단위
                
                vehicle_type = arrive.iloc[0].type
                parking_times.append({
                    "id": v_id,
                    "type": vehicle_type,
                    "duration": duration
                })
        
        if parking_times:
            parking_df = pd.DataFrame(parking_times)
            
            # 히스토그램
            plt.hist(
                [
                    parking_df[parking_df.type == "normal"].duration,
                    parking_df[parking_df.type == "ev"].duration
                ],
                bins=10,
                label=["일반 차량", "전기차"]
            )
            plt.title('차량 유형별 주차 시간 분포')
            plt.xlabel('주차 시간 (시)')
            plt.ylabel('차량 수')
            plt.legend()
            plt.savefig(self.parking_duration_graph_path)
        
        # 3. 전기차 충전 패턴 분석
        charge_df = self.get_charge_dataframe()
        if not charge_df.empty:
            plt.figure(figsize=(10, 6))
            
            # 각 전기차별 충전 패턴 시각화
            for vehicle_id in charge_df.id.unique():
                v_data = charge_df[charge_df.id == vehicle_id]
                v_data = v_data.sort_values('time')
                if 'battery' in v_data.columns and not v_data.battery.isna().all():
                    plt.plot(v_data.time / 60, v_data.battery, marker='o', 
                            label=f'EV #{vehicle_id}')
            
            plt.title('전기차 충전 패턴')
            plt.xlabel('시간 (분)')
            plt.ylabel('배터리 (%)')
            plt.grid(True)
            plt.legend()
            plt.savefig(self.charge_graph_path)
        
        print(f"그래프가 생성되었습니다: {self.arrivals_graph_path}, {self.parking_duration_graph_path}")
        if not charge_df.empty:
            print(f"충전 그래프: {self.charge_graph_path}") 