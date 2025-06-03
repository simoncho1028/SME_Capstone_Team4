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
from typing import List

# 한글 폰트 설정
if platform.system() == 'Windows':
    plt.rcParams['font.family'] = 'Malgun Gothic'  # 윈도우 한글 폰트
elif platform.system() == 'Darwin':  # macOS
    plt.rcParams['font.family'] = 'AppleGothic'    # 맥OS 한글 폰트
else:  # Linux
    plt.rcParams['font.family'] = 'NanumGothic'    # 리눅스 한글 폰트

mpl.rcParams['axes.unicode_minus'] = False   # 마이너스 기호 깨짐 방지

from src.config import (
    SEED, SIM_TIME, PARKING_MAP,  # PARKING_MAP 추가
    generate_adjacent_charger_layouts  # 새로 추가한 함수도 import
)
from src.models.simulation import ParkingSimulation, CustomParkingSimulation
from src.utils.visualizer import ParkingVisualizer


def parse_arguments():
    """
    커맨드 라인 인자를 파싱합니다.
    
    Returns:
        파싱된 인자
    """
    parser = argparse.ArgumentParser(
        description="주차장 시뮬레이션",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--seed", 
        type=int, 
        default=42,
        help="난수 생성기 시드"
    )
    
    parser.add_argument(
        "--time", 
        type=int, 
        default=86400,
        help="시뮬레이션 시간 (초)"
    )
    
    parser.add_argument(
        "--normal", 
        type=int, 
        default=25,
        help="일반 차량 수"
    )
    
    parser.add_argument(
        "--ev", 
        type=int, 
        default=5,
        help="전기차 수"
    )
    
    parser.add_argument(
        "--parking-capacity", 
        type=int, 
        default=28,
        help="일반 주차면 용량"
    )
    
    parser.add_argument(
        "--charger-capacity", 
        type=int, 
        default=4,
        help="EV 충전소 용량"
    )
    
    parser.add_argument(
        "--visualize", 
        action="store_true",
        help="시뮬레이션 결과 시각화"
    )
    
    parser.add_argument(
        "--no-save-csv", 
        action="store_true",
        help="CSV 파일 저장 비활성화"
    )
    
    parser.add_argument(
        "--output-prefix", 
        type=str, 
        default="results_sim",
        help="출력 파일 이름 접두사"
    )

    # 애니메이션 관련 옵션 추가
    parser.add_argument(
        "--animation",
        action="store_true",
        help="시뮬레이션 결과 애니메이션 생성"
    )

    parser.add_argument(
        "--fps",
        type=int,
        default=5,  # 기본값을 5로 변경 (이전 수정사항 반영)
        help="애니메이션 프레임 속도 (FPS)"
    )

    parser.add_argument(
        "--dpi",
        type=int,
        default=80,  # 기본값을 80으로 변경 (이전 수정사항 반영)
        help="애니메이션 해상도 (DPI)"
    )

    parser.add_argument(
        "--speed",
        type=float,
        default=60.0,
        help="시뮬레이션 속도 (실제 1초당 시뮬레이션 시간 초)"
    )

    # 최적화 모드 추가
    parser.add_argument(
        "--optimize",
        action="store_true",
        help="충전소 배치 최적화 수행"
    )

    # 레이아웃 시각화 옵션 추가
    parser.add_argument(
        "--visualize-layout",
        action="store_true",
        help="각 층별 주차장 레이아웃을 이미지로 저장"
    )
    
    parser.add_argument(
        "--layout-dpi",
        type=int,
        default=100,
        help="레이아웃 이미지 해상도 (DPI)"
    )
    
    parser.add_argument(
        "--layout-cell-size",
        type=float,
        default=1.0,
        help="레이아웃 그리드 셀 크기"
    )
    
    parser.add_argument(
        "--layout-font-size",
        type=float,
        default=8.0,
        help="레이아웃 텍스트 폰트 크기"
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
    global PARKING_MAP
    
    # 기본 맵에서 충전소 제거 (모든 'C'를 'P'로 변경)
    base_map = [row.replace('C', 'P') for row in PARKING_MAP]
    
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
        
        # 임시로 PARKING_MAP 수정
        original_map = PARKING_MAP
        PARKING_MAP = layout
        
        # 시뮬레이션 실행
        sim = CustomParkingSimulation(
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
        PARKING_MAP = original_map
    
    print("\n\n[INFO] 시뮬레이션 완료!")
    return best_layout, best_metrics


def main():
    """
    메인 실행 함수
    """
    global PARKING_MAP
    
    # 인자 파싱
    args = parse_arguments()
    
    # 시드 설정
    random.seed(args.seed)
    
    # 결과 저장 디렉토리 생성
    output_dir = create_output_directory(args.output_prefix)
    
    # 레이아웃 시각화 요청이 있는 경우
    if args.visualize_layout:
        from parking_layout_visualizer import ParkingLayoutVisualizer
        visualizer = ParkingLayoutVisualizer("layout_images")
        visualizer.dpi = args.layout_dpi
        visualizer.cell_size = args.layout_cell_size
        visualizer.font_size = args.layout_font_size
        visualizer.visualize_all_floors()
        if not args.animation:  # 애니메이션을 생성하지 않는 경우 종료
            return

    # 최적화 모드
    if args.optimize:
        print("[INFO] 인접한 2개 충전소의 최적 배치를 찾습니다...")
        best_layout, best_metrics = optimize_adjacent_chargers(args)
        
        print("\n=== 최적화 결과 ===")
        print(f"총 비용: {best_metrics['total_cost']:,} 원")
        print(f"충전소 공실률: {best_metrics['idle_rate'] * 100:.2f} %")
        print(f"충전 실패율: {best_metrics['charge_fail_rate'] * 100:.2f} %")
        print(f"주차 실패율: {best_metrics['parking_fail_rate'] * 100:.2f} %")
        
        # 최적 배치로 PARKING_MAP 업데이트
        PARKING_MAP = best_layout
        print("\n[INFO] 최적 배치로 시뮬레이션을 실행합니다...")
    else:
        print(f"[INFO] 일반 시뮬레이션을 시작합니다...")
        print(f"[INFO] 설정: seed={args.seed}, 시간={args.time}초, 일반차량={args.normal}, EV={args.ev}")
    
    # 시뮬레이션 실행
    sim = CustomParkingSimulation(
        parking_capacity=args.parking_capacity,
        charger_capacity=args.charger_capacity,
        sim_time=args.time,
        random_seed=args.seed,
        normal_count=args.normal,
        ev_count=args.ev
    )
    sim.run()
    
    # 결과 요약 출력
    print("\n=== 시뮬레이션 결과 ===")
    sim.print_summary()
    
    # 최적화/운영 지표 출력
    print("\n--- 최적화/운영 지표 ---")
    charger_count = args.charger_capacity
    sim_time = args.time
    total_cost = sim.logger.calculate_charger_cost(charger_count)
    idle_rate = sim.logger.calculate_charger_idle_rate(sim_time, charger_count)
    charge_fail_rate = sim.logger.calculate_charge_fail_rate()
    parking_fail_rate = sim.logger.calculate_parking_fail_rate()
    print(f"충전소 설치+유지 총비용: {total_cost:,} 원")
    print(f"충전소 공실률: {idle_rate * 100:.2f} %")
    print(f"충전 실패율: {charge_fail_rate * 100:.2f} %")
    print(f"주차 실패율: {parking_fail_rate * 100:.2f} %")
    
    # 최종 주차장 배치 시각화
    print("\n최종 주차장 배치:")
    visualize_layout_terminal(PARKING_MAP, "최종 배치")
    
    # 결과 CSV 저장
    if not args.no_save_csv:
        csv_path = os.path.join(output_dir, "simulation_log.csv")
        sim.logger.save_to_csv(csv_path)
        print(f"\n[INFO] 결과가 {csv_path}에 저장되었습니다.")
    
    # 그래프 생성 및 저장
    arrivals_path = os.path.join(output_dir, "arrivals_by_hour.png")
    parking_path = os.path.join(output_dir, "parking_duration.png")
    charging_path = os.path.join(output_dir, "charging_patterns.png")
    
    sim.logger.set_graph_paths(arrivals_path, parking_path, charging_path)
    sim.generate_plots()
    
    # 시각화 (마지막에 한 번만)
    if args.visualize:
        print("\n[INFO] 최종 주차장 상태 시각화 중...")
        logger = sim.get_results()
        df = logger.get_dataframe()
        visualizer = ParkingVisualizer()
        frames = visualizer.generate_animation_data(df)
        
        if len(frames) > 0:
            # 최종 상태만 저장
            visualizer.save_state_image(
                frames[-1]['occupied'], 
                frames[-1]['charging'], 
                os.path.join(output_dir, "final_parking_state.png"),
                f"최종 주차장 상태 (시간: {frames[-1]['time']/3600:.1f}시간)"
            )
            print(f"[INFO] 최종 상태 이미지가 {output_dir} 디렉토리에 저장되었습니다.")
    
    # 애니메이션 (요청된 경우에만)
    if args.animation:
        print("\n[INFO] 주차장 상태 애니메이션 생성 중...")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        animation_filename = f"parking_animation_{timestamp}.mp4"
        animation_path = os.path.join(output_dir, animation_filename)
        
        from parking_animation import prepare_animation_data, animate_parking
        logger = sim.get_results()
        df = logger.get_dataframe()
        
        print("[INFO] 애니메이션 데이터 준비 중...")
        frames = prepare_animation_data(df, args.speed)
        
        if frames:
            print(f"[INFO] 애니메이션 생성 중... (FPS: {args.fps}, DPI: {args.dpi})")
            animate_parking(frames, animation_path, args.fps, args.dpi)
            print(f"[INFO] 애니메이션이 저장되었습니다: {animation_path}")
        else:
            print("[ERROR] 애니메이션 프레임을 생성할 수 없습니다.")
            return 1
    
    print("\n[INFO] 시뮬레이션이 완료되었습니다.")
    print(f"[INFO] 모든 결과가 {output_dir} 디렉토리에 저장되었습니다.")
    return 0


if __name__ == "__main__":
    sys.exit(main()) 