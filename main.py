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
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
import platform
from typing import List, Dict
import simpy
import json
from collections import defaultdict

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
from src.utils.visualizer import ParkingVisualizer
from src.models.vehicle import Vehicle
from parking_animation import prepare_frames_from_log, ParkingAnimationVisualizer, COLORS, FLOORS, PARKING_MAPS
from parking_layout_visualizer import ParkingLayoutVisualizer

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
        # 배터리 레벨 랜덤 생성 (전기차만)
        battery_level = None
        if info["type"].lower() == "ev":
            battery_level = random.uniform(20.0, 80.0)
        
        vehicle = Vehicle(
            vehicle_id=info["id"],
            vehicle_type=info["type"].lower(),
            arrival_time=0,  # 초기값 0으로 설정 (실제 도착 시간은 plan_daily_entries에서 결정)
            building_id=info["building"],
            battery_level=battery_level
        )
        vehicles.append(vehicle)
    
    return vehicles

def parse_arguments():
    """
    커맨드 라인 인자를 파싱합니다.
    
    Returns:
        파싱된 인자
    """
    parser = argparse.ArgumentParser(
        description="주차장 시뮬레이션 - EV 충전소 최적화",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # 시뮬레이션 기본 설정
    sim_group = parser.add_argument_group('시뮬레이션 설정')
    sim_group.add_argument(
        "--seed", 
        type=int, 
        default=42,
        help="난수 생성기 시드"
    )
    
    sim_group.add_argument(
        "--time", 
        type=int, 
        default=86400,
        help="시뮬레이션 시간 (초)"
    )
    
    # 차량 설정
    vehicle_group = parser.add_argument_group('차량 설정')
    vehicle_group.add_argument(
        "--normal", 
        type=int, 
        default=830,  # 일반 차량 수 증가
        help="일반 차량 수"
    )
    
    vehicle_group.add_argument(
        "--ev", 
        type=int, 
        default=36,   # 전기차 수 증가
        help="전기차 수"
    )
    
    # 주차장 설정
    parking_group = parser.add_argument_group('주차장 설정')
    parking_group.add_argument(
        "--total-capacity",
        type=int,
        default=686,  # 전체 주차면 수 (일반차량 + 전기차)
        help="전체 주차면 용량"
    )
    
    parking_group.add_argument(
        "--ev-chargers",
        type=int,
        default=36,    # 기본 충전소 수
        help="설치할 EV 충전소 수"
    )
    
    # 결과 저장 설정
    output_group = parser.add_argument_group('결과 저장 설정')
    output_group.add_argument(
        "--no-save-csv", 
        action="store_true",
        help="CSV 파일 저장 비활성화"
    )
    
    output_group.add_argument(
        "--output-prefix", 
        type=str, 
        default="results_sim",
        help="출력 파일 이름 접두사"
    )
    
    # 시각화 설정
    viz_group = parser.add_argument_group('시각화 설정')
    viz_group.add_argument(
        "--visualize", 
        action="store_true",
        help="시뮬레이션 결과 시각화"
    )
    
    viz_group.add_argument(
        "--visualize-layout",
        action="store_true",
        help="주차장 레이아웃 시각화"
    )
    
    viz_group.add_argument(
        "--layout-dpi",
        type=int,
        default=100,
        help="레이아웃 이미지 해상도 (DPI)"
    )
    
    viz_group.add_argument(
        "--layout-cell-size",
        type=float,
        default=1.0,
        help="레이아웃 셀 크기"
    )
    
    viz_group.add_argument(
        "--layout-font-size",
        type=int,
        default=10,
        help="레이아웃 폰트 크기"
    )
    
    viz_group.add_argument(
        "--animation",
        action="store_true",
        help="시뮬레이션 결과 애니메이션 생성"
    )
    
    viz_group.add_argument(
        "--fps",
        type=int,
        default=5,
        help="애니메이션 프레임 속도 (FPS)"
    )
    
    viz_group.add_argument(
        "--dpi",
        type=int,
        default=100,
        help="이미지 해상도 (DPI)"
    )
    
    viz_group.add_argument(
        "--speed",
        type=float,
        default=60.0,
        help="시뮬레이션 속도 (실제 1초당 시뮬레이션 시간 초)"
    )
    
    # 최적화 설정
    opt_group = parser.add_argument_group('최적화 설정')
    opt_group.add_argument(
        "--optimize",
        action="store_true",
        help="충전소 배치 최적화 수행"
    )
    
    opt_group.add_argument(
        "--optimization-trials",
        type=int,
        default=10,
        help="최적화 시도 횟수"
    )
    
    opt_group.add_argument(
        "--min-chargers",
        type=int,
        default=4,
        help="최소 충전소 수"
    )
    
    opt_group.add_argument(
        "--max-chargers",
        type=int,
        default=12,
        help="최대 충전소 수"
    )

    return parser.parse_args()


def create_output_directory(prefix):
    """
    결과 파일을 저장할 디렉토리를 생성합니다.
    
    Args:
        prefix: 디렉토리 이름 접두사
        
    Returns:
        생성된 디렉토리 경로
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
    global PARKING_MAPS
    
    # 인자 파싱
    args = parse_arguments()
    
    # 시드 설정
    random.seed(args.seed)
    
    # 결과 저장 디렉토리 설정
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f"../results_sim_{timestamp}"  # 'results_' 중복 제거
    os.makedirs(results_dir, exist_ok=True)
    print(f"[INFO] 결과 저장 디렉토리 생성: {results_dir}")
    
    # 로거 초기화 (파일 경로 지정)
    logger = SimulationLogger(
        log_file=os.path.join(results_dir, "simulation_log.csv"),
        stats_file=os.path.join(results_dir, "simulation_stats.json")
    )
    
    # 주차장 관리자 초기화
    parking_manager = ParkingManager()
    
    # 충전소 할당
    parking_manager.allocate_chargers(args.ev_chargers)
    
    # 레이아웃 시각화 요청이 있는 경우
    if args.visualize_layout:
        visualizer = ParkingLayoutVisualizer(output_dir=results_dir)
        
        # 충전소 위치 수집 (parking_manager에서 할당된 위치 사용)
        charger_positions = defaultdict(list)
        for floor, row, col in parking_manager.ev_chargers:
            charger_positions[floor].append((row, col))
        
        # 충전소 위치 설정 및 시각화
        visualizer.set_charger_positions(charger_positions)
        visualizer.visualize_all_floors()
        
        print(f"\n[INFO] 주차장 레이아웃 및 충전소 위치가 '{results_dir}' 디렉토리에 저장되었습니다.")

    # 최적화 모드
    if args.optimize:
        print(f"[INFO] EV 충전소 최적 배치 탐색을 시작합니다...")
        print(f"[INFO] 충전소 수 범위: {args.min_chargers}~{args.max_chargers}개")
        print(f"[INFO] 각 설정당 시도 횟수: {args.optimization_trials}회")
        
        best_metrics = None
        best_charger_count = None
        
        # 충전소 수를 변경해가며 최적화
        for charger_count in range(args.min_chargers, args.max_chargers + 1):
            print(f"\n[INFO] 충전소 {charger_count}개 배치에 대한 시뮬레이션 시작...")
            
            # 여러 번 시도하여 평균 성능 측정
            trial_metrics = []
            for trial in range(args.optimization_trials):
                print(f"  - 시도 {trial + 1}/{args.optimization_trials}")
                
                sim = ParkingSimulation(
                    parking_capacity=args.total_capacity - charger_count,
                    charger_capacity=charger_count,
                    sim_time=args.time,
                    random_seed=args.seed + trial,
                    normal_count=args.normal,
                    ev_count=args.ev
                )
                sim.run()
                
                # 성능 지표 계산
                metrics = {
                    'total_cost': sim.logger.calculate_charger_cost(charger_count),
                    'idle_rate': sim.logger.calculate_charger_idle_rate(args.time, charger_count),
                    'charge_fail_rate': sim.logger.calculate_charge_fail_rate(),
                    'parking_fail_rate': sim.logger.calculate_parking_fail_rate()
                }
                trial_metrics.append(metrics)
            
            # 평균 성능 계산
            avg_metrics = {
                key: sum(m[key] for m in trial_metrics) / len(trial_metrics)
                for key in trial_metrics[0].keys()
            }
            
            # 목적 함수 계산 (비용과 실패율의 가중 합)
            objective = (
                avg_metrics['total_cost'] * 0.4 +
                avg_metrics['idle_rate'] * 0.2 * 1000000 +
                avg_metrics['charge_fail_rate'] * 0.2 * 1000000 +
                avg_metrics['parking_fail_rate'] * 0.2 * 1000000
            )
            
            print(f"\n[INFO] 충전소 {charger_count}개 배치 결과:")
            print(f"  - 총 비용: {avg_metrics['total_cost']:,}원")
            print(f"  - 충전소 공실률: {avg_metrics['idle_rate']*100:.2f}%")
            print(f"  - 충전 실패율: {avg_metrics['charge_fail_rate']*100:.2f}%")
            print(f"  - 주차 실패율: {avg_metrics['parking_fail_rate']*100:.2f}%")
            
            # 더 나은 결과를 찾았다면 업데이트
            if best_metrics is None or objective < best_metrics['objective']:
                best_metrics = avg_metrics
                best_metrics['objective'] = objective
                best_charger_count = charger_count
        
        print(f"\n[INFO] 최적화 완료!")
        print(f"[INFO] 최적 충전소 수: {best_charger_count}개")
        print(f"[INFO] 최적 성능 지표:")
        print(f"  - 총 비용: {best_metrics['total_cost']:,}원")
        print(f"  - 충전소 공실률: {best_metrics['idle_rate']*100:.2f}%")
        print(f"  - 충전 실패율: {best_metrics['charge_fail_rate']*100:.2f}%")
        print(f"  - 주차 실패율: {best_metrics['parking_fail_rate']*100:.2f}%")
        
        # 최적 설정으로 마지막 시뮬레이션 실행
        args.ev_chargers = best_charger_count
    
    # 일반 시뮬레이션 실행
    print(f"\n[INFO] 시뮬레이션을 시작합니다...")
    print(f"[INFO] 설정:")
    print(f"  - 일반 차량: {args.normal}대")
    print(f"  - 전기차: {args.ev}대")
    print(f"  - 충전소: {args.ev_chargers}개")
    print(f"  - 총 주차면: {args.total_capacity}면")
    
    # SimPy 환경 초기화
    env = simpy.Environment()
    
    # 시뮬레이션 객체 생성
    sim = ParkingSimulation(
        env=env,
        parking_manager=parking_manager,
        logger=logger
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
        sim.print_summary()  # 파일 저장 없이 출력만
    
    # 최적화/운영 지표 출력
    print("\n--- 최적화/운영 지표 ---")
    total_cost = sim.logger.calculate_charger_cost(args.ev_chargers)
    idle_rate = sim.logger.calculate_charger_idle_rate(args.time, args.ev_chargers)
    charge_fail_rate = sim.logger.calculate_charge_fail_rate()
    parking_fail_rate = sim.logger.calculate_parking_fail_rate()
    
    print(f"충전소 설치+유지 총비용: {total_cost:,} 원")
    print(f"충전소 공실률: {idle_rate * 100:.2f} %")
    print(f"충전 실패율: {charge_fail_rate * 100:.2f} %")
    print(f"주차 실패율: {parking_fail_rate * 100:.2f} %")
    
    # 결과 CSV 저장
    if not args.no_save_csv:
        csv_path = os.path.join(results_dir, "simulation_log.csv")
        sim.logger.save_to_csv(csv_path)
        print(f"\n[INFO] 결과가 {csv_path}에 저장되었습니다.")
        
        # 시뮬레이션 요약 저장
        sim.save_summary_to_file(results_dir)
    
    # 그래프 생성 및 저장
    arrivals_path = os.path.join(results_dir, "arrivals_by_hour.png")
    charging_path = os.path.join(results_dir, "charging_patterns.png")
    
    sim.logger.set_graph_paths(arrivals_path, charging_path)
    sim.generate_plots()
    
    # 시각화 (마지막에 한 번만)
    if args.visualize:
        print("\n[INFO] 최종 주차장 상태 시각화 중...")
        df = sim.logger.get_dataframe()
        visualizer = ParkingVisualizer()
        frames = visualizer.generate_animation_data(df)
        
        if len(frames) > 0:
            # 최종 상태만 저장
            visualizer.save_state_image(
                frames[-1]['occupied'], 
                frames[-1]['charging'], 
                os.path.join(results_dir, "final_parking_state.png"),
                f"최종 주차장 상태 (시간: {frames[-1]['time']/3600:.1f}시간)"
            )
            print(f"[INFO] 최종 상태 이미지가 {results_dir} 디렉토리에 저장되었습니다.")
    
    # 애니메이션 (요청된 경우에만)
    if args.animation:
        print("\n[INFO] 4층 통합 주차장 상태 애니메이션 생성 중...")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        animation_filename = f"parking_animation_4floors_{timestamp}.mp4"
        animation_path = os.path.join(results_dir, animation_filename)
        
        df = sim.logger.get_dataframe()
        
        print("[INFO] 4층 통합 애니메이션 데이터 준비 중...")
        frames = prepare_frames_from_log(df, frame_interval=30.0)
        
        if frames:
            print(f"[INFO] 4층 통합 애니메이션 생성 중... (FPS: {args.fps}, DPI: {args.dpi})")
            visualizer = ParkingAnimationVisualizer(FLOORS, PARKING_MAPS, COLORS)
            visualizer.animate(frames, animation_path, fps=args.fps, dpi=args.dpi)
            print(f"[INFO] 4층 통합 애니메이션이 저장되었습니다: {animation_path}")
        else:
            print("[ERROR] 애니메이션 프레임을 생성할 수 없습니다.")
            return 1
    
    print("\n[INFO] 시뮬레이션이 완료되었습니다.")
    print(f"[INFO] 모든 결과가 {results_dir} 디렉토리에 저장되었습니다.")
    return 0


if __name__ == "__main__":
    sys.exit(main()) 