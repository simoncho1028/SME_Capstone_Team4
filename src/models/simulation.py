"""
주차장 시뮬레이션을 실행하고 관리하는 모듈입니다.
"""
import simpy
import random
from typing import Optional, Callable, Dict, Any

from src.config import SEED, SIM_TIME
from src.utils.logger import SimulationLogger
from src.models.generator import VehicleGenerator, CustomVehicleGenerator


class ParkingSimulation:
    """
    주차장 시뮬레이션 환경을 관리하는 클래스
    """
    
    def __init__(self, 
                 parking_capacity: int = 28, 
                 charger_capacity: int = 4,
                 sim_time: float = SIM_TIME,
                 random_seed: int = SEED):
        """
        시뮬레이션 환경을 초기화합니다.
        
        Args:
            parking_capacity: 일반 주차 공간 수
            charger_capacity: EV 충전 공간 수
            sim_time: 시뮬레이션 실행 시간 (초 단위)
            random_seed: 난수 생성을 위한 시드
        """
        # 난수 시드 설정
        random.seed(random_seed)
        
        # SimPy 환경 초기화
        self.env = simpy.Environment()
        
        # 리소스 초기화
        self.parking_res = simpy.Resource(self.env, capacity=parking_capacity)
        self.charger_res = simpy.Resource(self.env, capacity=charger_capacity)
        
        # 로거 초기화
        self.logger = SimulationLogger()
        
        # 시뮬레이션 시간 설정
        self.sim_time = sim_time
        
        # 차량 생성기 초기화
        self.generator = VehicleGenerator(
            env=self.env,
            parking_res=self.parking_res,
            charger_res=self.charger_res,
            logger=self.logger
        )
    
    def move_vehicle(self, vehicle, new_pos):
        """
        차량을 새로운 위치로 이동시킵니다.
        
        Args:
            vehicle: 이동할 차량
            new_pos: 새로운 위치 (row, col)
        """
        # 차량 위치 업데이트
        vehicle.position = new_pos
        
        # 이동 이벤트 로깅
        self.logger.add_event(
            vehicle_id=vehicle.id,
            vehicle_type=vehicle.type,
            event='move',
            time=self.env.now,
            pos=new_pos,
            battery=vehicle.battery if vehicle.type == 'ev' else None
        )
    
    def run(self) -> None:
        """
        시뮬레이션을 실행합니다.
        """
        # 차량 생성 프로세스 시작
        self.env.process(self.generator.run())
        
        # 시뮬레이션 실행
        self.env.run(until=self.sim_time)
    
    def get_results(self) -> SimulationLogger:
        """
        시뮬레이션 결과 로거를 반환합니다.
        
        Returns:
            로깅된 이벤트 데이터
        """
        return self.logger
    
    def print_summary(self) -> None:
        """
        시뮬레이션 결과 요약을 출력합니다.
        """
        self.logger.print_summary()
    
    def generate_plots(self) -> None:
        """
        시뮬레이션 결과를 시각화하는 그래프를 생성합니다.
        """
        self.logger.generate_plots()
    
    def save_results(self, filename: Optional[str] = None) -> str:
        """
        시뮬레이션 결과를 CSV 파일로 저장합니다.
        
        Args:
            filename: 저장할 파일 이름 (없으면 타임스탬프로 자동 생성)
            
        Returns:
            저장된 파일 경로
        """
        return self.logger.save_to_csv(filename)


class CustomParkingSimulation(ParkingSimulation):
    """
    사용자 정의 파라미터로 주차장 시뮬레이션을 실행할 수 있는 확장된 클래스
    """
    
    def __init__(self, 
                 parking_capacity: int = 28, 
                 charger_capacity: int = 4,
                 sim_time: float = SIM_TIME,
                 random_seed: int = SEED,
                 normal_count: int = 25,
                 ev_count: int = 5,
                 interarrival_func: Optional[Callable[[], float]] = None,
                 parking_duration_func: Optional[Callable[[], float]] = None,
                 battery_level_func: Optional[Callable[[], float]] = None,
                 charge_time_func: Optional[Callable[[], float]] = None):
        """
        사용자 정의 파라미터로 시뮬레이션 환경을 초기화합니다.
        
        Args:
            parking_capacity: 일반 주차 공간 수
            charger_capacity: EV 충전 공간 수
            sim_time: 시뮬레이션 실행 시간 (초 단위)
            random_seed: 난수 생성을 위한 시드
            normal_count: 생성할 일반 차량 수
            ev_count: 생성할 전기차 수
            interarrival_func: 차량 도착 간격을 샘플링하는 함수 (이제 사용되지 않음)
            parking_duration_func: 주차 시간을 샘플링하는 함수
            battery_level_func: 배터리 잔량을 샘플링하는 함수
            charge_time_func: 충전 시간을 샘플링하는 함수
        """
        # 부모 클래스 초기화
        super().__init__(
            parking_capacity=parking_capacity,
            charger_capacity=charger_capacity,
            sim_time=sim_time,
            random_seed=random_seed
        )
        
        # 사용자 정의 생성기 생성
        self.generator = CustomVehicleGenerator(
            env=self.env,
            parking_res=self.parking_res,
            charger_res=self.charger_res,
            logger=self.logger,
            sim_time=self.sim_time,
            interarrival_func=interarrival_func,
            normal_count=normal_count,
            ev_count=ev_count
        )
        
        # 사용자 정의 샘플링 함수를 helpers.py의 함수들과 교체 (parking_duration만 해당)
        if parking_duration_func:
            import src.utils.helpers as helpers
            # CustomVehicleGenerator에서 주차 시간을 Vehicle 생성 시점에 샘플링하므로,
            # 여기서 sample_parking_duration 함수 자체를 교체하는 로직은 필요 없어짐.
            # 대신 Vehicle 클래스 내부에서 sample_time_dependent_parking_duration 함수가 env를 사용하도록 해야 함.
            # 하지만 현재 Vehicle 클래스는 이미 env를 가지고 있고 해당 함수에 넘겨주고 있으므로 추가 수정 필요 없음.
            pass # 여기서는 특별히 할 일 없음. parking_duration_func 인자는 CustomVehicleSimulation에서만 사용될 수 있음.

        # 사용자 정의 샘플링 함수를 helpers.py의 함수들과 교체
        if any([battery_level_func, charge_time_func]):
            if battery_level_func:
                helpers.sample_battery_level = battery_level_func
            
            if charge_time_func:
                helpers.sample_charge_time = charge_time_func 