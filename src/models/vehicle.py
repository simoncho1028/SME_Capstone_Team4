"""
주차장 내 차량을 나타내는 모델 클래스
"""
from typing import Tuple, Optional, Dict
import simpy
import random
from dataclasses import dataclass

from src.utils.helpers import (
    sample_battery_level, sample_normal_parking_duration,
    sample_charge_time, calculate_parking_time
)
from src.utils.logger import SimulationLogger

@dataclass
class Vehicle:
    """
    주차장 내 차량을 나타내는 클래스
    일반 차량 및 전기차 모두 이 클래스로 표현됩니다.
    """
    id: str  # 차량 고유 ID (N1-N830 또는 EV1-EV36)
    type: str  # "normal" 또는 "ev"
    state: str = "outside"  # "outside", "parked", "double_parked"
    next_entry_time: float = 0  # 다음 입차 예정 시각 (초)
    parking_duration: float = 0  # 현재 주차 예정 시간 (초)
    battery_level: Optional[float] = None  # 전기차의 경우 배터리 잔량 (0-100)
    
    def __post_init__(self):
        """초기화 이후 추가 설정"""
        if self.type == "ev":
            self.battery_level = sample_battery_level()
    
    def set_next_parking_duration(self):
        """다음 주차 시간을 설정"""
        self.parking_duration = sample_normal_parking_duration()
    
    def set_next_entry_time(self, current_time: float, time_window: float):
        """
        다음 입차 시각을 설정
        
        Args:
            current_time: 현재 시각 (초)
            time_window: 입차 가능 시간 범위 (초)
        """
        self.next_entry_time = current_time + time_window
    
    def __str__(self):
        return f"Vehicle(id={self.id}, type={self.type}, state={self.state})"

class Vehicle:
    """
    주차장 내 차량을 나타내는 클래스
    일반 차량 및 전기차 모두 이 클래스로 표현됩니다.
    """
    
    # 현재 주차된 차량의 위치를 추적
    occupied_spots: Dict[str, Tuple[int, int]] = {}
    
    def __init__(self, 
                 vehicle_info: Dict,
                 env: simpy.Environment,
                 parking_res: simpy.Resource, 
                 charger_res: simpy.Resource,
                 logger: SimulationLogger):
        """
        차량 객체를 초기화합니다.
        
        Args:
            vehicle_info: 차량 정보 딕셔너리 (id, type, building 포함)
            env: SimPy 환경 객체
            parking_res: 일반 주차면 리소스
            charger_res: EV 충전소 리소스
            logger: 이벤트 로깅을 위한 로거 객체
        """
        self.id = vehicle_info["id"]
        self.type = vehicle_info["type"]
        self.building = vehicle_info["building"]
        self.env = env
        self.parking_res = parking_res
        self.charger_res = charger_res
        self.logger = logger
        
        # 전기차만 배터리 속성 초기화 (0-100%)
        self.battery_capacity = 100.0 if self.type == "EV" else None
        self.battery = sample_battery_level() if self.type == "EV" else None
        
        # 차량 위치 초기화 (입구에서 시작)
        self.pos = (0, 0)  # 입구 위치를 (0, 0)으로 가정
        
        # 사용 중인 리소스 추적
        self.current_resource: Optional[simpy.Resource] = None
        self.resource_request: Optional[simpy.Request] = None
        
        # 주차 시간 초기화 (일반 차량은 정규 분포, 전기차는 기존 방식 유지)
        self.parking_duration = (
            sample_normal_parking_duration() if self.type == "Normal"
            else sample_charge_time()
        )
        
        # 차량 도착 이벤트 로깅
        self.log_event("arrive")
    
    def log_event(self, event: str) -> None:
        """
        차량 이벤트를 로그에 기록합니다.
        
        Args:
            event: 이벤트 유형
        """
        self.logger.add_event(
            vehicle_id=self.id,
            vehicle_type=self.type,
            building=self.building,
            event=event,
            time=self.env.now,
            pos=self.pos,
            battery=self.battery
        )
    
    def find_parking_spot(self) -> Optional[Tuple[int, int]]:
        """
        비어있는 주차면을 찾습니다.
        
        Returns:
            Optional[Tuple[int, int]]: 주차면 위치 또는 None
        """
        # 임시로 랜덤한 위치 반환
        while True:
            spot = (random.randint(1, 10), random.randint(1, 10))
            if spot not in Vehicle.occupied_spots.values():
                return spot
    
    def process(self):
        """
        차량의 전체 생애주기 프로세스
        주차→(충전)→출차 과정을 관리합니다.
        """
        # 1. 주차면 찾기
        spot = self.find_parking_spot()
        if not spot:
            print(f"[ERROR] 차량 {self.id} ({self.type}, {self.building}) 주차 실패")
            self.log_event("park_fail")
            return
        
        # 2. 리소스 요청
        if self.type == "EV" and self.battery < 100:
            self.resource_request = self.charger_res.request()
            yield self.resource_request
            self.current_resource = self.charger_res
        else:
            self.resource_request = self.parking_res.request()
            yield self.resource_request
            self.current_resource = self.parking_res
        
        # 3. 주차면까지 이동
        parking_time = calculate_parking_time(self.pos, spot)
        yield self.env.timeout(parking_time)
        
        # 4. 주차면 점유
        self.pos = spot
        Vehicle.occupied_spots[self.id] = spot
        self.log_event("park_start")
        
        # 5. 전기차 충전 (조건 충족 시)
        if self.type == "EV" and self.battery < 100 and self.current_resource == self.charger_res:
            self.log_event("charge_start")
            initial_battery = self.battery
            
            # 충전 진행
            start_time = self.env.now
            while self.env.now - start_time < self.parking_duration and self.battery < 100:
                # 배터리 잔량 업데이트 (선형적으로 증가)
                elapsed = self.env.now - start_time
                self.battery = min(100, initial_battery + (100 - initial_battery) * (elapsed / self.parking_duration))
                
                # 로그 기록
                self.log_event("charge_update")
                
                # 1분마다 상태 업데이트
                yield self.env.timeout(60)
            
            # 충전 완료
            self.battery = 100
            self.log_event("charge_end")
        else:
            # 일반 주차
            yield self.env.timeout(self.parking_duration)
        
        # 6. 출차
        if self.current_resource and self.resource_request:
            self.current_resource.release(self.resource_request)
            self.current_resource = None
            self.resource_request = None
            
            if self.id in Vehicle.occupied_spots:
                del Vehicle.occupied_spots[self.id]
            
            self.pos = (0, 0)  # 출구로 이동
            self.log_event("depart") 