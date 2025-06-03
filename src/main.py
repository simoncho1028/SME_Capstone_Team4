"""
주차장 시뮬레이션 메인 스크립트
"""
import simpy
from src.models.vehicle_manager import VehicleManager
from src.simulation.parking_simulation import ParkingSimulation

def main():
    # SimPy 환경 초기화
    env = simpy.Environment()
    
    # 차량 관리자 초기화 (일반차 830대, 전기차 36대)
    vehicle_manager = VehicleManager(normal_count=830, ev_count=36)
    
    # 시뮬레이션 초기화
    simulation = ParkingSimulation(
        env=env,
        vehicle_manager=vehicle_manager,
        max_parking_spots=800,
        max_charging_spots=36
    )
    
    # 시뮬레이션 실행 (48시간)
    simulation.run(simulation_time=48*60*60)
    
    # 결과 출력
    counts = vehicle_manager.get_vehicle_counts()
    print("\n=== 시뮬레이션 결과 ===")
    print(f"외부 차량: {counts['outside']}대")
    print(f"주차된 차량: {counts['parked']}대")
    print(f"이중주차 차량: {counts['double_parked']}대")

if __name__ == "__main__":
    main() 