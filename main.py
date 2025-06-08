#!/usr/bin/env python3
"""
아파트 주차장 EV 충전 시뮬레이션 메인 실행 파일

사용법:
    python main.py


이 파일은 주차장 시뮬레이션을 실행하고 결과를 시각화합니다.
"""
import random
import sys
import argparse
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
import platform
from typing import List, Dict
import simpy
import json
from collections import defaultdict
from src.utils.helpers import sample_battery_level

# 한글 폰트 설정
if platform.system() == 'Windows':
    plt.rcParams['font.family'] = 'Malgun Gothic'  # 윈도우 한글 폰트
elif platform.system() == 'Darwin':  # macOS
    plt.rcParams['font.family'] = 'AppleGothic'    # 맥OS 한글 폰트
else:  # Linux
    plt.rcParams['font.family'] = 'NanumGothic'    # 리눅스 한글 폰트

mpl.rcParams['axes.unicode_minus'] = False   # 마이너스 기호 깨짐 방지

from src.config import (
    SEED, SIM_TIME, PARKING_MAPS,
    generate_adjacent_charger_layouts,
    ENTRY_RATIO, get_arrival_time,
    CELL_PARK, CELL_CHARGER
)
from src.simulation.parking_simulation import ParkingSimulation
from src.models.parking_manager import ParkingManager
from src.utils.logger import SimulationLogger
from src.models.vehicle import Vehicle
from parking_layout_visualizer import ParkingLayoutVisualizer
from src.utils.parking_animation import run_animation

def load_vehicles() -> Dict:
    """vehicles.json 파일에서 차량 데이터를 로드합니다."""
    with open("data/vehicles.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["vehicles"]

def create_vehicles(env: simpy.Environment, vehicle_data: Dict) -> List[Vehicle]:
    """
    차량 객체 리스트를 생성합니다.
    
    Args:
        env: SimPy 환경
        vehicle_data: vehicles.json에서 로드한 차량 데이터
        
    Returns:
        Vehicle 객체 리스트
    """
    vehicles = []
    # 입차할 차량 수 계산
    total_vehicles = len(vehicle_data)
    
    # 모든 차량 생성
    for vehicle_id, info in vehicle_data.items():
        # 배터리 레벨 샘플링 (전기차만)
        battery_level = None
        if info["type"].lower() == "ev":
            battery_level = sample_battery_level()
        
        vehicle = Vehicle(
            vehicle_id=info["id"],
            vehicle_type=info["type"].lower(),
            arrival_time=0,  # 초기값 0으로 설정 (실제 도착 시간은 plan_daily_entries에서 결정)
            building_id=info["building"],
            battery_level=battery_level
        )
        vehicles.append(vehicle)
    
    return vehicles

def parse_args():
    parser = argparse.ArgumentParser(description='주차장 시뮬레이션')
    parser.add_argument("--no-save-csv", action="store_true", help="CSV 저장 안 함")
    parser.add_argument("--visualize-layout", action="store_true", help="주차장 레이아웃 시각화")
    parser.add_argument("--animation", action="store_true", help="주차장 상태 애니메이션 실행")
    parser.add_argument("--save-video", action="store_true", help="애니메이션을 MP4로 저장")
    parser.add_argument("--animation-speed", type=float, default=0.1, help="애니메이션 속도 (초)")
    parser.add_argument("--layout", type=str, default='config/layout.json', help='레이아웃 파일 경로')
    parser.add_argument("--log", type=str, help='시뮬레이션 로그 파일 경로')
    parser.add_argument("--time", type=int, default=86400, help="시뮬레이션 시간 (초, 기본값: 24시간)")
    parser.add_argument("--normal", type=int, default=830, help="일반 차량 수 (기본값: 830)")
    parser.add_argument("--ev", type=int, default=36, help="전기차 수 (기본값: 36)")
    parser.add_argument("--parking-capacity", type=int, default=866, help="전체 주차면 수 (기본값: 866)")
    parser.add_argument("--seed", type=int, default=42, help="랜덤 시드 (기본값: 42)")
    parser.add_argument('--charger', type=int, default=36, help='충전소 개수 (기본값: 36)')
    parser.add_argument('--ratio', nargs=2, type=float, default=[1.0, 1.3],
                        help='입차 비율 범위 (예: --ratio 1.0 1.3)')
    return parser.parse_args()


def create_output_directory(prefix):
    """
    결과 파일을 저장할 디렉토리를 생성합니다.
    
    Args:
        prefix: 디렉토리 이름 접두사
        
    Returns:
 p       생성된 디렉토리 경로
    """
    # 현재 시간을 포함한 디렉토리명 생성
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join("..", f"results_{prefix}_{timestamp}")
    
    # 디렉토리가 존재하지 않으면 생성
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"[INFO] 결과 저장 디렉토리 생성: {output_dir}")
    
    return output_dir


def visualize_layout_terminal(layout: List[str], title: str = ""):
    """
    주차장 배치를 터미널에 컬러로 시각화합니다.
    """
    # Windows에서 ANSI 색상 지원 활성화
    if platform.system() == 'Windows':
        import ctypes
        kernel32 = ctypes.windll.kernel32
        kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), 7)
    
    # ANSI 색상 코드
    colors = {
        'N': '\033[90m',  # 회색 (경계/미사용)
        'E': '\033[93m',  # 노란색 (입구/출구)
        'R': '\033[37m',  # 흰색 (도로)
        'P': '\033[94m',  # 파란색 (일반 주차면)
        'C': '\033[92m',  # 초록색 (충전소)
        'RESET': '\033[0m'
    }
    
    print("\n")  # 빈 줄 추가
    if title:
        print(f"=== {title} ===")
        print()
    
    # 상단 경계선
    width = len(layout[0]) * 4 + 1
    print("+" + "-" * width + "+")
    
    # 맵 출력
    for row in layout:
        print("|", end=" ")
        for cell in row:
            color = colors.get(cell, '')
            reset = colors['RESET']
            if cell == 'C':
                print(f"{color}[{cell}]{reset}", end=" ")
            else:
                print(f"{color} {cell} {reset}", end=" ")
        print("|")
    
    # 하단 경계선
    print("+" + "-" * width + "+")
    print()
    
    # 범례 출력
    print("범례:")
    for cell, desc in [('N', '경계/미사용'), ('E', '입구/출구'), ('R', '도로'),
                      ('P', '일반 주차면'), ('C', '충전소')]:
        color = colors.get(cell, '')
        reset = colors['RESET']
        if cell == 'C':
            print(f"{color}[{cell}]{reset}: {desc}  ", end="")
        else:
            print(f"{color} {cell} {reset}: {desc}  ", end="")
    print("\n")


def optimize_adjacent_chargers(args):
    """
    인접한 2개의 충전소에 대한 최적의 배치를 찾습니다.
    """
    global PARKING_MAPS
    
    # 기본 맵에서 충전소 제거 (모든 'C'를 'P'로 변경)
    base_map = [row.replace('C', 'P') for row in PARKING_MAPS]
    
    # 가능한 모든 인접 충전소 배치 생성
    layouts = generate_adjacent_charger_layouts(base_map)
    
    best_layout = None
    best_metrics = {
        'total_cost': float('inf'),
        'idle_rate': 1.0,
        'charge_fail_rate': 1.0,
        'parking_fail_rate': 1.0
    }
    
    print(f"[INFO] 총 {len(layouts)}개의 가능한 배치에 대해 시뮬레이션을 실행합니다...")
    print("\n진행률:")
    
    # 각 배치에 대해 시뮬레이션 실행
    for i, layout in enumerate(layouts, 1):
        print(f"\r[{i}/{len(layouts)}] {i/len(layouts)*100:.1f}% 완료", end="")
        
        # 임시로 PARKING_MAPS 수정
        original_map = PARKING_MAPS
        PARKING_MAPS = layout
        
        # 시뮬레이션 실행
        sim = ParkingSimulation(
            parking_capacity=args.parking_capacity - 2,  # 충전소 2개만큼 감소
            charger_capacity=2,
            sim_time=args.time,
            random_seed=args.seed,
            normal_count=args.normal,
            ev_count=args.ev
        )
        sim.run()
        
        # 성능 지표 계산
        total_cost = sim.logger.calculate_charger_cost(2)  # 2개 충전소
        idle_rate = sim.logger.calculate_charger_idle_rate(args.time, 2)
        charge_fail_rate = sim.logger.calculate_charge_fail_rate()
        parking_fail_rate = sim.logger.calculate_parking_fail_rate()
        
        # 목적 함수 계산
        objective = (
            total_cost * 0.4 +
            idle_rate * 0.2 * 1000000 +
            charge_fail_rate * 0.2 * 1000000 +
            parking_fail_rate * 0.2 * 1000000
        )
        
        # 더 나은 결과를 찾았다면 업데이트
        if objective < best_metrics['total_cost']:
            best_layout = layout
            best_metrics = {
                'total_cost': total_cost,
                'idle_rate': idle_rate,
                'charge_fail_rate': charge_fail_rate,
                'parking_fail_rate': parking_fail_rate
            }
        
        # 원래 맵으로 복구
        PARKING_MAPS = original_map
    
    print("\n\n[INFO] 시뮬레이션 완료!")
    return best_layout, best_metrics


def main():
    """
    메인 실행 함수
    """
    args = parse_args()
    
    # 시드 설정
    random.seed(args.seed)
    
    # 결과 저장 디렉토리 설정
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f"../results_sim_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)
    print(f"[INFO] 결과 저장 디렉토리 생성: {results_dir}")
    
    # 로거 초기화
    logger = SimulationLogger(
        log_file=os.path.join(results_dir, "simulation_log.csv"),
        stats_file=os.path.join(results_dir, "simulation_stats.json")
    )
    
    # 주차장 관리자 초기화
    parking_manager = ParkingManager()
    
    # 충전소 할당
    parking_manager.allocate_chargers(args.charger)
    
    # 충전소 위치를 JSON으로 저장
    charger_positions = defaultdict(list)
    for floor, row, col in parking_manager.ev_chargers:
        charger_positions[floor].append([row, col])
    
    charger_positions_file = os.path.join(results_dir, "charger_positions.json")
    with open(charger_positions_file, "w", encoding="utf-8") as f:
        json.dump(dict(charger_positions), f, indent=2, ensure_ascii=False)
    print(f"[INFO] 충전소 위치가 {charger_positions_file}에 저장되었습니다.")
    
    # 레이아웃 시각화 요청이 있는 경우
    if args.visualize_layout:
        visualizer = ParkingLayoutVisualizer(output_dir=results_dir)
        
        # 충전소 위치 수집 (parking_manager에서 할당된 위치 사용)
        charger_positions = defaultdict(list)
        for floor, row, col in parking_manager.ev_chargers:
            charger_positions[floor].append((row, col))
        
        # 충전소 위치를 JSON 파일로 저장
        charger_positions_file = os.path.join(results_dir, "charger_positions.json")
        with open(charger_positions_file, 'w', encoding='utf-8') as f:
            json.dump(dict(charger_positions), f, indent=2, ensure_ascii=False)
        print(f"\n[INFO] 충전소 위치 정보가 {charger_positions_file}에 저장되었습니다.")
        
        # 충전소 위치 설정 및 시각화
        visualizer.set_charger_positions(charger_positions)
        visualizer.visualize_all_floors()
        
        print(f"\n[INFO] 주차장 레이아웃 및 충전소 위치가 '{results_dir}' 디렉토리에 저장되었습니다.")
    
    # 시뮬레이션 설정 출력
    print("\n=== 시뮬레이션 설정 ===")
    print(f"  - 일반 차량: {args.normal}대")
    print(f"  - 전기차: {args.ev}대")
    print(f"  - 충전소: {args.charger}개")
    print(f"  - 총 주차면: {args.parking_capacity}면")
    
    # SimPy 환경 초기화
    env = simpy.Environment()
    
    # 시뮬레이션 객체 생성
    total_vehicles = args.normal + args.ev
    sim = ParkingSimulation(
        env=env,
        parking_manager=parking_manager,
        logger=logger,
        total_vehicle_count=total_vehicles,
        normal_count=args.normal,
        ev_count=args.ev,
        ratio_min=args.ratio[0],
        ratio_max=args.ratio[1]
    )
    
    # 차량 데이터 로드 및 차량 생성
    vehicle_data = load_vehicles()
    vehicles = create_vehicles(env, vehicle_data)
    
    # 차량들을 시뮬레이션의 outside_vehicles에 추가
    for vehicle in vehicles:
        sim.outside_vehicles[vehicle.vehicle_id] = vehicle
    
    # 시뮬레이션 실행
    sim.run(until=args.time)
    
    # 결과 요약 출력
    print("\n=== 시뮬레이션 결과 ===")
    if not args.no_save_csv:
        sim.print_summary(results_dir)  # results_dir 전달
    else:
        sim.print_summary(".")  # 현재 디렉토리 사용
    
    # 결과 CSV 저장
    if not args.no_save_csv:
        csv_path = os.path.join(results_dir, "simulation_log.csv")
        sim.logger.save_to_csv(csv_path)
        print(f"\n[INFO] 결과가 {csv_path}에 저장되었습니다.")
        
        # 시뮬레이션 요약 저장 (중복 제거)
        # sim.save_summary_to_file(results_dir)  # 이미 print_summary에서 저장됨
    
    # 애니메이션 실행
    if args.animation:
        print("\n주차장 상태 애니메이션을 시작합니다...")
        log_file = os.path.join(results_dir, "simulation_log.csv")
        
        # 충전소 위치 정보 수집
        charger_positions = defaultdict(list)
        for floor, row, col in parking_manager.ev_chargers:
            # 층 이름을 원래 형식으로 변환
            if floor == 'GF':
                original_floor = 'Ground'
            elif floor == 'B1F':
                original_floor = 'B1'
            elif floor == 'B2F':
                original_floor = 'B2'
            elif floor == 'B3F':
                original_floor = 'B3'
            else:
                original_floor = floor
            charger_positions[original_floor].append((row, col))
        
        print(f"[INFO] 충전소 위치 정보:")
        for floor in sorted(charger_positions.keys()):
            print(f"  - {floor}: {len(charger_positions[floor])}개")
            print(f"    위치: {charger_positions[floor]}")
        
        run_animation("json", log_file, args.animation_speed, args.save_video, charger_positions)
    
    return 0


if __name__ == "__main__":
    sys.exit(main()) 