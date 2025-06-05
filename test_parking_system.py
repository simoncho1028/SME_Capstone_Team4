import json
from parking_system import ParkingSystem

def load_vehicles():
    """vehicles.json 파일에서 차량 정보를 로드합니다."""
    with open('data/vehicles.json', 'r', encoding='utf-8') as f:
        return json.load(f)["vehicles"]

def main():
    # 주차 시스템 초기화
    parking_system = ParkingSystem()
    
    # 차량 정보 로드
    vehicles = load_vehicles()
    
    # 테스트를 위해 각 동별로 하나씩 차량을 선택
    test_vehicles = {}
    for vehicle in vehicles.values():
        building = vehicle["building"]
        if building not in test_vehicles:
            test_vehicles[building] = vehicle
        if len(test_vehicles) == 8:  # 8개 동 모두 선택되면 중단
            break
    
    print("=== 주차 배정 테스트 시작 ===")
    print("\n1. 각 동별 테스트 차량:")
    for building, vehicle in test_vehicles.items():
        print(f"{building}: 차량 ID - {vehicle['id']}")
    
    print("\n2. 주차 배정 결과:")
    assignments = {}
    for vehicle in test_vehicles.values():
        result = parking_system.assign_parking_spot(vehicle)
        assignments[vehicle["building"]] = result
        print(f"\n{vehicle['building']} 소속 차량 {vehicle['id']}의 주차 배정:")
        if result:
            spot = result["assigned_spot"]
            print(f"- 배정된 층: {spot['floor']}")
            print(f"- 위치: [{spot['x']}, {spot['y']}]")
            print(f"- 동에서의 거리: {spot['distance']}")
        else:
            print("- 주차 가능한 공간을 찾을 수 없습니다.")
    
    print("\n3. 주차장 상태:")
    status = parking_system.get_parking_status()
    for floor, info in status.items():
        print(f"\n{floor}:")
        print(f"- 전체 주차면: {info['total_spots']}")
        print(f"- 사용 중인 주차면: {info['occupied_spots']}")
        print(f"- 남은 주차면: {info['total_spots'] - info['occupied_spots']}")

if __name__ == "__main__":
    main() 