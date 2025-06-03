"""
주차장 내 차량을 나타내는 모델 클래스
"""
from typing import Optional
from dataclasses import dataclass

@dataclass
class Vehicle:
    """주차장 내 차량을 나타내는 클래스"""
    vehicle_id: str  # 차량 고유 ID
    vehicle_type: str  # "normal" 또는 "ev"
    arrival_time: float  # 도착 예정 시간
    building_id: str = "A"  # 차량이 속한 동 번호
    state: str = "outside"  # "outside", "parked", "double_parked"
    battery_level: Optional[float] = None  # 전기차의 경우 배터리 잔량 (0-100)
    parking_duration: float = 3600.0  # 주차 예정 시간 (초)

    def __post_init__(self):
        """초기화 이후 추가 설정"""
        if self.vehicle_type not in ["normal", "ev"]:
            raise ValueError("vehicle_type must be either 'normal' or 'ev'")
        
        if self.vehicle_type == "ev" and self.battery_level is None:
            self.battery_level = 100.0  # 기본값 설정
        
        if self.state not in ["outside", "parked", "double_parked"]:
            raise ValueError("Invalid vehicle state")

    def run(self, sim) -> None:
        """차량의 주차장 이용 프로세스"""
        # 도착 시간까지 대기
        yield sim.env.timeout(self.arrival_time)
        
        # 입차 처리
        sim.handle_vehicle_entry(self)
        
        # 주차 시간 동안 대기
        yield sim.env.timeout(self.parking_duration)
        
        # 출차 처리
        sim.handle_vehicle_exit(self)

    def needs_charging(self) -> bool:
        """전기차의 충전 필요 여부 확인"""
        return self.vehicle_type == "ev" and self.battery_level < 100.0

    def update_state(self, new_state: str) -> None:
        """차량 상태 업데이트"""
        if new_state not in ["outside", "parked", "double_parked"]:
            raise ValueError("Invalid vehicle state")
        self.state = new_state

    def start_charging(self) -> None:
        """충전 시작"""
        if self.vehicle_type != "ev":
            raise ValueError("Only EV can start charging")
        self.state = "charging"

    def update_battery(self, elapsed_time: float) -> None:
        """배터리 잔량 업데이트"""
        if self.vehicle_type != "ev":
            return
            
        if self.state == "charging" and self.battery_level < 100.0:
            # 선형적으로 배터리 증가 (1시간당 50% 충전 가정)
            charge_rate = 50.0 / 3600  # %/초
            self.battery_level = min(100.0, self.battery_level + charge_rate * elapsed_time) 