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

# 한글 폰트 설정
if platform.system() == 'Windows':
    plt.rcParams['font.family'] = 'Malgun Gothic'  # 윈도우 한글 폰트
elif platform.system() == 'Darwin':  # macOS
    plt.rcParams['font.family'] = 'AppleGothic'    # 맥OS 한글 폰트
else:  # Linux
    plt.rcParams['font.family'] = 'NanumGothic'    # 리눅스 한글 폰트

mpl.rcParams['axes.unicode_minus'] = False   # 마이너스 기호 깨짐 방지

from src.config import SEED, SIM_TIME
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
        default=7200,
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


def main():
    """
    메인 실행 함수
    """
    # 인자 파싱
    args = parse_arguments()
    
    # 시드 설정
    random.seed(args.seed)
    
    print(f"[INFO] 시뮬레이션을 시작합니다...")
    print(f"[INFO] 설정: seed={args.seed}, 시간={args.time}초, 일반차량={args.normal}, EV={args.ev}")
    
    # 결과 저장 디렉토리 생성
    output_dir = create_output_directory(args.output_prefix)
    
    # 시뮬레이션 생성 및 실행
    sim = CustomParkingSimulation(
        parking_capacity=args.parking_capacity,
        charger_capacity=args.charger_capacity,
        sim_time=args.time,
        random_seed=args.seed,
        normal_count=args.normal,
        ev_count=args.ev
    )
    
    # 시뮬레이션 실행
    sim.run()
    
    # 결과 요약 출력
    print("\n=== 시뮬레이션 결과 ===")
    sim.print_summary()
    
    # 결과 CSV 저장 (기본적으로 활성화)
    if not args.no_save_csv:
        csv_path = os.path.join(output_dir, "simulation_log.csv")
        sim.logger.save_to_csv(csv_path)
        print(f"\n[INFO] 결과가 {csv_path}에 저장되었습니다.")
    
    # 그래프 생성 및 저장
    arrivals_path = os.path.join(output_dir, "arrivals_by_hour.png")
    parking_path = os.path.join(output_dir, "parking_duration.png")
    charging_path = os.path.join(output_dir, "charging_patterns.png")
    
    # 생성된 그래프 파일 이름을 설정하고 로거에 전달
    sim.logger.set_graph_paths(arrivals_path, parking_path, charging_path)
    sim.generate_plots()
    
    # 시각화
    if args.visualize:
        print("\n[INFO] 주차장 상태 시각화 중...")
        
        # 로그 데이터 가져오기
        logger = sim.get_results()
        df = logger.get_dataframe()
        
        # 시각화 도구 초기화
        visualizer = ParkingVisualizer()
        
        # 주차장 상태 변화 데이터 생성
        frames = visualizer.generate_animation_data(df)
        
        # 주요 상태 스냅샷 저장 (시작, 중간, 끝)
        if len(frames) > 0:
            # 첫 번째 프레임
            visualizer.save_state_image(
                frames[0]['occupied'], 
                frames[0]['charging'], 
                os.path.join(output_dir, "parking_state_start.png"),
                f"주차장 상태 (시작: {frames[0]['time']/3600:.1f}시간)"
            )
            
            # 중간 프레임 (있다면)
            if len(frames) > 2:
                mid_idx = len(frames) // 2
                visualizer.save_state_image(
                    frames[mid_idx]['occupied'], 
                    frames[mid_idx]['charging'], 
                    os.path.join(output_dir, "parking_state_middle.png"),
                    f"주차장 상태 (중간: {frames[mid_idx]['time']/3600:.1f}시간)"
                )
            
            # 마지막 프레임
            visualizer.save_state_image(
                frames[-1]['occupied'], 
                frames[-1]['charging'], 
                os.path.join(output_dir, "parking_state_end.png"),
                f"주차장 상태 (종료: {frames[-1]['time']/3600:.1f}시간)"
            )
            
            print(f"[INFO] 시각화 이미지가 {output_dir} 디렉토리에 저장되었습니다.")

    # 애니메이션 생성
    if args.animation:
        print("\n[INFO] 주차장 상태 애니메이션 생성 중...")
        
        # 현재 시간을 파일명에 추가
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        animation_filename = f"parking_animation_{timestamp}.mp4"
        animation_path = os.path.join(output_dir, animation_filename)
        
        # 애니메이션 생성
        from parking_animation import prepare_animation_data, animate_parking
        
        # 로그 데이터 가져오기
        logger = sim.get_results()
        df = logger.get_dataframe()
        
        # 애니메이션 데이터 준비
        print("[INFO] 애니메이션 데이터 준비 중...")
        frames = prepare_animation_data(df, args.speed)
        
        if frames:
            # 애니메이션 생성 및 저장
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