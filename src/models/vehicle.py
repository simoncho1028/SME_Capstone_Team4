"""
주차장 내 차량을 나타내는 모델 클래스
"""
from typing import Optional
from dataclasses import dataclass
from src.utils.helpers import sample_parking_duration
import numpy as np
import random
import os

# GMM 컴포넌트 정의
GMM_COMPONENTS = [
    (0.359, 108, 42),   # (weight, mean, std)
    (0.180, 213, 22),
    (0.310, 403, 101),
    (0.151, 635, 111)
]

# 충전 속도 설정 (아이오닉5 기준)
CHARGING_RATE_PER_MINUTE = 0.162  # % per minute

@dataclass
class Vehicle:
    """주차장 내 차량을 나타내는 클래스"""
    vehicle_id: str  # 차량 고유 ID (N123 또는 E123 형식)
    vehicle_type: str  # "normal" 또는 "ev"
    arrival_time: float  # 도착 예정 시간
    building_id: str  # 차량이 속한 동 번호 (예: "1동")
    state: str = "outside"  # "outside", "parked", "double_parked", "charging"
    battery_level: Optional[float] = None  # 전기차의 경우 배터리 잔량 (0-100)
    parking_duration: Optional[float] = None  # 주차 예정 시간 (초)
    assigned_charging_time: Optional[float] = None  # 할당된 충전 시간 (분)
    is_charging: bool = False  # 충전 상태
    finished_charging: bool = False  # 충전 완료 여부
    charging_start_time: Optional[float] = None  # 충전 시작 시간
    charged_amount: float = 0.0  # 충전된 양 (%)
    env: Optional[object] = None  # SimPy 환경
    parking_manager: Optional[object] = None  # 주차장 관리자

    def __post_init__(self):
        """초기화 이후 추가 설정"""
        if self.vehicle_type not in ["normal", "ev"]:
            raise ValueError("vehicle_type must be either 'normal' or 'ev'")
        
        if self.vehicle_type == "ev":
            # 초기 배터리 레벨 설정 (20-80% 사이)
            self.battery_level = random.uniform(20, 80) if self.battery_level is None else self.battery_level
            
            # GMM에서 충전 시간 샘플링
            self.assigned_charging_time = self._sample_charging_time_from_gmm()
        
        if self.state not in ["outside", "parked", "double_parked", "charging"]:
            raise ValueError("Invalid vehicle state")
            
        # 주차 시간이 설정되지 않은 경우, 도착 시각에 따른 감마 분포에서 샘플링
        if self.parking_duration is None:
            self.parking_duration = sample_parking_duration(self.arrival_time)

    def _sample_charging_time_from_gmm(self) -> float:
        """GMM에서 충전 시간을 샘플링합니다."""
        # 컴포넌트 선택
        weights = [comp[0] for comp in GMM_COMPONENTS]
        selected_comp = np.random.choice(len(GMM_COMPONENTS), p=weights)
        
        # 선택된 컴포넌트에서 시간 샘플링
        _, mean, std = GMM_COMPONENTS[selected_comp]
        sampled_time = np.random.normal(mean, std)
        
        # 음수 방지
        return max(0, sampled_time)

    def process(self):
        """
        SimPy 환경에서 차량 프로세스를 실행합니다.
        """
        return self.run()
    
    def run(self):
        """차량의 주차장 이용 프로세스"""
        # 도착 시간까지 대기 (현재 시뮬레이션 시간에서 도착 시간까지의 차이)
        if self.arrival_time > self.env.now:
            wait_time = self.arrival_time - self.env.now
            yield self.env.timeout(wait_time)
        
        # 입차 처리
        self.parking_manager.handle_vehicle_entry(self)
        
        # 주차 성공 여부 확인
        if self.vehicle_id in self.parking_manager.parked_vehicles:
            # 주차 시간 동안 대기
            yield self.env.timeout(self.parking_duration)
            
            # 출차 처리
            self.parking_manager.handle_vehicle_exit(self)
            
            # 차량 상태를 outside로 변경
            self.update_state("outside")
        else:
            print(f"[TEST] 차량 {self.vehicle_id} 주차 실패")

    def needs_charging(self) -> bool:
        """전기차의 충전 필요 여부 확인"""
        return (self.vehicle_type == "ev" and 
                self.battery_level < 80 and 
                not self.finished_charging)

    def update_state(self, new_state: str) -> None:
        """차량 상태 업데이트"""
        if new_state not in ["outside", "parked", "double_parked", "charging"]:
            raise ValueError("Invalid vehicle state")
        self.state = new_state
        if new_state == "outside":
            self.stop_charging()

    def start_charging(self, current_time: float) -> None:
        """충전 시작"""
        if self.vehicle_type != "ev":
            raise ValueError("Only EV can start charging")
        self.state = "charging"
        self.is_charging = True
        self.charging_start_time = current_time
        self.charged_amount = 0.0

    def stop_charging(self) -> None:
        """충전 종료"""
        self.state = "parked"
        self.is_charging = False
        self.charging_start_time = None

    def update_charging(self, current_time: float) -> None:
        """충전 상태 업데이트"""
        if not self.is_charging or self.charging_start_time is None:
            return
            
        # 충전 경과 시간 계산 (분 단위)
        elapsed_minutes = (current_time - self.charging_start_time) / 60
        
        # 할당된 충전 시간이 지났는지 확인
        if elapsed_minutes >= self.assigned_charging_time:
            # 총 충전된 양 계산
            total_charged = CHARGING_RATE_PER_MINUTE * self.assigned_charging_time
            self.battery_level = min(100, self.battery_level + total_charged)
            self.finished_charging = True
            self.stop_charging()
            return
            
        # 현재까지 충전된 양 계산
        current_charged = CHARGING_RATE_PER_MINUTE * elapsed_minutes
        self.battery_level = min(100, self.battery_level + current_charged - self.charged_amount)
        self.charged_amount = current_charged

    def __str__(self) -> str:
        """문자열 표현"""
        status = f"Vehicle(id={self.vehicle_id}, type={self.vehicle_type}, " \
                f"building={self.building_id}, state={self.state}"
        if self.vehicle_type == "ev":
            status += f", battery={self.battery_level:.1f}%, charging={self.is_charging}"
            if self.assigned_charging_time is not None:
                status += f", assigned_charging_time={self.assigned_charging_time:.1f}min"
        status += ")"
        return status 

    def park(self, floor: str, position: tuple, parking_manager) -> None:
        """차량을 주차 처리"""
        if parking_manager.park_vehicle(self):
            # 주차 성공 로그 기록
            parking_manager.logger.log_event(
                time=parking_manager.env.now,
                vehicle_id=self.vehicle_id,
                event="park_success",
                floor=floor,
                pos=position,
                battery=self.battery_level,
                parking_duration=self.parking_duration  # 주차 시간 정보 추가
            )
            
            # 통계 업데이트
            parking_manager.logger.update_stats("park_success", self.parking_duration)
            
            # 전기차이고 충전이 필요한 경우 충전 시작
            if self.vehicle_type == "ev" and self.needs_charging():
                self.start_charging(parking_manager.env.now)
                parking_manager.logger.log_event(
                    time=parking_manager.env.now,
                    vehicle_id=self.vehicle_id,
                    event="charge_start",
                    floor=floor,
                    pos=position,
                    battery=self.battery_level
                )
        else:
            # 주차 실패 로그 기록
            parking_manager.logger.log_event(
                time=parking_manager.env.now,
                vehicle_id=self.vehicle_id,
                event="park_fail"
            ) 