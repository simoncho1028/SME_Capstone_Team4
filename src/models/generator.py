"""
차량 생성 및 도착 프로세스를 관리하는 모듈입니다.
"""
import simpy
import random
import numpy as np
from typing import List, Callable, Optional, Dict
from pathlib import Path
import json

from src.config import NUM_NORMAL, NUM_EV
from src.utils.helpers import (
    sample_interarrival, sample_time_dependent_interarrival, sample_interarrival_time
)
from src.utils.logger import SimulationLogger
from src.models.vehicle import Vehicle
from src.utils.charger_allocator import ChargerAllocator


class VehicleGenerator:
    """
    시뮬레이션 동안 차량을 생성하는 클래스입니다.
    """
    
    def __init__(self, 
                 env: simpy.Environment, 
                 parking_res: simpy.Resource, 
                 charger_res: simpy.Resource,
                 logger: SimulationLogger,
                 charger_allocator: ChargerAllocator,
                 data_path: str = "data"):
        """
        차량 생성기를 초기화합니다.
        
        Args:
            env: SimPy 환경 객체
            parking_res: 일반 주차면 리소스
            charger_res: EV 충전소 리소스
            logger: 이벤트 로깅을 위한 로거 객체
            charger_allocator: 충전소 위치 관리자
            data_path: 차량 데이터 파일 경로
        """
        self.env = env
        self.parking_res = parking_res
        self.charger_res = charger_res
        self.logger = logger
        self.charger_allocator = charger_allocator
        self.data_path = Path(data_path)
        
        # 차량 데이터 로드
        self.vehicles = self._load_vehicles()
        self.vehicle_ids = list(self.vehicles.keys())
        random.shuffle(self.vehicle_ids)  # 무작위 순서로 차량 생성
    
    def _load_vehicles(self) -> Dict:
        """
        차량 데이터를 JSON 파일에서 로드합니다.
        
        Returns:
            Dict: 차량 정보 딕셔너리
        """
        vehicles_file = self.data_path / "vehicles.json"
        if not vehicles_file.exists():
            raise FileNotFoundError(f"차량 데이터 파일을 찾을 수 없습니다: {vehicles_file}")
        
        with open(vehicles_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def generate_vehicle(self, vtype: str) -> None:
        """
        주어진 유형의 차량을 생성하고 시뮬레이션에 추가합니다.
        
        Args:
            vtype: 차량 유형 ("normal" 또는 "ev")
        """
        vehicle = Vehicle(
            vid=self.next_id,
            vtype=vtype,
            env=self.env,
            parking_res=self.parking_res,
            charger_res=self.charger_res,
            logger=self.logger,
            charger_allocator=self.charger_allocator
        )
        
        self.next_id += 1
        self.env.process(vehicle.process())
    
    def run(self) -> None:
        """
        지정된 수의 일반 차량과 전기차를 생성하는 프로세스를 시작합니다.
        차량은 지수 분포에 따라 도착합니다.
        """
        # 모든 차량(일반 + 전기차)을 랜덤 순서로 생성
        total_vehicles = self.normal_count + self.ev_count
        vehicle_types = ["normal"] * self.normal_count + ["ev"] * self.ev_count
        random.shuffle(vehicle_types)  # 차량 유형 순서를 랜덤하게 섞음
        
        # 랜덤 순서로 차량 생성
        for vtype in vehicle_types:
            yield self.env.timeout(self.interarrival_func())
            self.generate_vehicle(vtype)


class CustomVehicleGenerator(VehicleGenerator):
    """
    사용자 정의 분포나 도착 패턴을 적용할 수 있는 확장된 차량 생성기입니다.
    """
    
    def __init__(self, 
                 env: simpy.Environment, 
                 parking_res: simpy.Resource, 
                 charger_res: simpy.Resource,
                 logger: SimulationLogger,
                 charger_allocator: ChargerAllocator,
                 sim_time: float,
                 interarrival_func: Optional[Callable[[], float]] = None,
                 normal_count: int = NUM_NORMAL,
                 ev_count: int = NUM_EV,
                 data_path: str = "data"):
        """
        사용자 정의 차량 생성기를 초기화합니다.
        
        Args:
            env: SimPy 환경 객체
            parking_res: 일반 주차면 리소스
            charger_res: EV 충전소 리소스
            logger: 이벤트 로깅을 위한 로거 객체
            charger_allocator: 충전소 위치 관리자
            sim_time: 전체 시뮬레이션 시간 (초)
            interarrival_func: 도착 시간 간격을 샘플링하는 함수 (이제 사용되지 않음)
            normal_count: 생성할 일반 차량 수
            ev_count: 생성할 전기차 수
            data_path: 차량 데이터 파일 경로
        """
        super().__init__(env, parking_res, charger_res, logger, charger_allocator, data_path)
        self.sim_time = sim_time
        self.normal_count = normal_count
        self.ev_count = ev_count
        # 전체 차량 수는 일반차 수 + 전기차 수
        self.total_vehicles = normal_count + ev_count
        self.env = env
        # 차량 유형 리스트를 전체 차량 수에 맞게 생성 및 섞기
        self.vehicle_types = ["normal"] * normal_count + ["ev"] * ev_count
        random.shuffle(self.vehicle_types)
        # 현실적인 입차 시각 계산 (시뮬레이션 시작 시간 0부터)
        self.entry_times = self._generate_realistic_entry_times(self.total_vehicles)

    def _generate_realistic_entry_times(self, total_vehicles):
        """
        현실적인 입차 시각을 시간대별 정규화된 비율에 따라 샘플링합니다.
        24시간 단위로 반복되는 패턴을 시뮬레이션 전체 시간에 맞게 적용합니다.
        """
        from src.utils.helpers import normalized_entry_ratios
        total_day_seconds = 86400  # 24시간(초)
        ratios = np.array(normalized_entry_ratios)
        
        # 전체 시뮬레이션 기간 동안의 일수 계산
        total_days = int(np.ceil(self.sim_time / total_day_seconds))
        
        entry_times = []
        for day in range(total_days):
            # 해당 일자의 시작 시간
            day_start = day * total_day_seconds
            
            # 하루 단위 차량 수 계산 (마지막 날은 남은 시간 비율만큼만)
            if day == total_days - 1 and self.sim_time % total_day_seconds != 0:
                day_ratio = (self.sim_time % total_day_seconds) / total_day_seconds
                day_vehicles = int(total_vehicles * day_ratio / total_days)
            else:
                day_vehicles = int(total_vehicles / total_days)
            
            # 시간대별 목표 차량 수 (해당 일자의 총 차량 수 * 비율)
            entries_per_hour = np.round(day_vehicles * ratios).astype(int)
            
            # 각 시간대별로 차량 생성 시간 샘플링
            for hour, n in enumerate(entries_per_hour):
                if n > 0:
                    # 해당 시간대 내에서 무작위 시각 샘플링 후 일자 시작 시간 더하기
                    hour_entries = hour * 3600 + np.random.uniform(0, 3600, n)
                    entry_times.extend(day_start + hour_entries)
        
        # 샘플링된 입차 시각 정렬
        entry_times.sort()
        
        # 총 차량 수에 맞게 조정
        if len(entry_times) < total_vehicles:
            print(f"[WARN] 샘플링된 입차 시각({len(entry_times)}개)이 총 차량 수({total_vehicles}개)보다 적습니다.")
        elif len(entry_times) > total_vehicles:
            entry_times = entry_times[:total_vehicles]
        
        return entry_times

    def run(self) -> None:
        """
        현실적인 입차 시각에 따라 차량을 생성합니다.
        """
        # 입차 시각과 차량 유형을 zip하여 순서대로 차량 생성
        for i, (vtype, entry_time) in enumerate(zip(self.vehicle_types, self.entry_times)):
            # 현재 env.now에서 entry_time까지 대기
            # 시뮬레이션 시작 시간(env.now)보다 이전 입차 시각은 바로 생성
            wait_time = max(0, entry_time - self.env.now)
            
            # 차량 도착 예정 시간
            arrival_time = self.env.now + wait_time
            
            # 시뮬레이션 종료 시간 이전에 도착하는 차량만 생성
            if arrival_time < self.sim_time:
                 yield self.env.timeout(wait_time)
                 # 차량 ID는 생성 순서대로 부여
                 self.generate_vehicle(vtype)
            else:
                 # 시뮬레이션 종료 시간 이후 도착 예정 차량은 생성하지 않음
                 # print(f"[DEBUG] 차량 {i} ({vtype}) 시뮬레이션 종료 시간({self.sim_time:.2f}) 이후 도착 예정({arrival_time:.2f}). 생성 스킵.")
                 pass 