#!/usr/bin/env python3
"""
주차장 시뮬레이션 애니메이션 시각화 스크립트

사용법:
    python parking_animation.py <simulation_log.csv> [output.mp4]

이 스크립트는 시뮬레이션 로그를 읽어 주차장 상태를 애니메이션으로 보여줍니다.
"""
import sys
import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
from typing import List, Dict, Any, Tuple, Optional
import platform

from src.config import PARKING_MAP
from src.utils.visualizer import ParkingVisualizer

# 한글 폰트 설정
if platform.system() == 'Windows':
    plt.rcParams['font.family'] = 'Malgun Gothic'  # 윈도우 한글 폰트
elif platform.system() == 'Darwin':  # macOS
    plt.rcParams['font.family'] = 'AppleGothic'    # 맥OS 한글 폰트
else:  # Linux
    plt.rcParams['font.family'] = 'NanumGothic'    # 리눅스 한글 폰트

plt.rcParams['axes.unicode_minus'] = False   # 마이너스 기호 깨짐 방지


def parse_arguments():
    """
    커맨드 라인 인자를 파싱합니다.
    
    Returns:
        파싱된 인자
    """
    parser = argparse.ArgumentParser(
        description="주차장 시뮬레이션 애니메이션 시각화",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "log_file", 
        type=str,
        help="시뮬레이션 로그 CSV 파일 경로"
    )
    
    parser.add_argument(
        "output_file", 
        type=str, 
        nargs='?',
        default="parking_animation.mp4",
        help="저장할 애니메이션 파일 경로 (mp4)"
    )
    
    parser.add_argument(
        "--fps", 
        type=int, 
        default=10,
        help="애니메이션 프레임 속도 (FPS)"
    )
    
    parser.add_argument(
        "--dpi", 
        type=int, 
        default=100,
        help="이미지 해상도 (DPI)"
    )
    
    parser.add_argument(
        "--speed", 
        type=float, 
        default=60.0,
        help="시뮬레이션 속도 (실제 1초당 시뮬레이션 시간 초)"
    )
    
    return parser.parse_args()


def prepare_animation_data(log_file: str, speed_factor: float = 60.0) -> List[Dict[str, Any]]:
    """
    시뮬레이션 로그 파일로부터 애니메이션 프레임 데이터를 생성합니다.
    
    Args:
        log_file: 로그 CSV 파일 경로
        speed_factor: 시뮬레이션 속도 배율 (실제 1초당 몇 초의 시뮬레이션 시간인지)
        
    Returns:
        애니메이션 프레임 데이터 목록
    """
    # 로그 데이터 읽기
    df = pd.read_csv(log_file)
    
    # 시간 기준 정렬
    df = df.sort_values('time')
    
    # 주차장 상태 변화 추적을 위한 데이터 구조
    frames = []
    
    # 현재 점유된 주차면 및 충전소 추적
    occupied_spots = set()
    charging_spots = set()
    moving_spots = set()  # 이동 중인 도로 셀 추적
    
    # 차량 ID별 위치 추적
    vehicle_positions = {}
    vehicle_types = {}
    vehicle_states = {}  # 차량 상태 추적 (moving, parked, charging)
    
    # 이전 프레임 시간
    prev_time = 0.0
    
    # 애니메이션 프레임 간격 (초 단위)
    frame_interval = speed_factor / 10  # 10fps 기준
    
    # 시간 간격마다 프레임 생성
    current_time = 0.0
    max_time = df['time'].max()
    
    # 모든 시간대 순회하며 프레임 생성
    while current_time <= max_time:
        # 현재 시간까지의 이벤트 필터링
        events_until_now = df[df.time <= current_time]
        
        # 이전 프레임의 이동 중인 셀 초기화
        moving_spots.clear()
        
        # 마지막 이벤트까지 처리
        for _, row in events_until_now.iterrows():
            if row['time'] <= prev_time:
                continue  # 이미 처리한 이벤트는 건너뜀
                
            event = row['event']
            vehicle_id = row['id']
            pos = (row['pos_r'], row['pos_c'])
            vehicle_type = row['type']
            
            # 차량 정보 업데이트
            vehicle_positions[vehicle_id] = pos
            vehicle_types[vehicle_id] = vehicle_type
            
            # 이벤트에 따라 주차장 상태 업데이트
            if event == 'park_start':
                cell_type = PARKING_MAP[pos[0]][pos[1]]
                if cell_type == 'P':
                    occupied_spots.add(pos)
                    vehicle_states[vehicle_id] = 'parked'
                elif cell_type == 'C':
                    charging_spots.add(pos)
                    vehicle_states[vehicle_id] = 'charging'
            
            elif event == 'charge_start':
                if pos in occupied_spots:
                    occupied_spots.remove(pos)
                charging_spots.add(pos)
                vehicle_states[vehicle_id] = 'charging'
            
            elif event == 'depart':
                if pos in occupied_spots:
                    occupied_spots.remove(pos)
                if pos in charging_spots:
                    charging_spots.remove(pos)
                if vehicle_id in vehicle_positions:
                    del vehicle_positions[vehicle_id]
                if vehicle_id in vehicle_types:
                    del vehicle_types[vehicle_id]
                if vehicle_id in vehicle_states:
                    del vehicle_states[vehicle_id]
            
            # 차량이 도로 위에 있고 주차/충전 중이 아닌 경우
            if PARKING_MAP[pos[0]][pos[1]] == 'R' and vehicle_id not in vehicle_states:
                vehicle_states[vehicle_id] = 'moving'
        
        # 현재 프레임의 차량 상태에 따라 셀 표시
        for vehicle_id, pos in vehicle_positions.items():
            if vehicle_id in vehicle_states:
                state = vehicle_states[vehicle_id]
                if state == 'moving' and PARKING_MAP[pos[0]][pos[1]] == 'R':
                    moving_spots.add(pos)
        
        # 프레임 추가
        frames.append({
            'time': current_time,
            'occupied': list(occupied_spots),
            'charging': list(charging_spots),
            'moving': list(moving_spots),
            'vehicles': vehicle_positions.copy(),
            'vehicle_types': vehicle_types.copy()
        })
        
        # 다음 시간 프레임으로 이동
        prev_time = current_time
        current_time += frame_interval
    
    return frames


def animate_parking(frames: List[Dict[str, Any]], output_file: str, fps: int = 10, dpi: int = 100):
    """
    주차장 상태 변화를 애니메이션으로 생성합니다.
    
    Args:
        frames: 애니메이션 프레임 데이터
        output_file: 저장할 파일 경로
        fps: 초당 프레임 수
        dpi: 해상도 DPI
    """
    if not frames:
        print("[ERROR] 애니메이션 프레임 데이터가 없습니다.")
        return
    
    # 시각화 도구 초기화
    visualizer = ParkingVisualizer()
    
    # 그래프 설정
    fig, ax1 = plt.subplots(figsize=(10, 8))
    
    # 색상 매핑
    cmap = mcolors.ListedColormap([visualizer.CELL_COLORS[c] for c in 'NERPCOCUM'])
    bounds = range(len('NERPCOCUM') + 1)
    norm = mcolors.BoundaryNorm(bounds, cmap.N)
    
    # 주차장 그리드 초기화
    cell_values = np.zeros(visualizer.grid.shape, dtype=int)
    img = ax1.imshow(cell_values, cmap=cmap, norm=norm)
    
    # 좌표축 설정
    ax1.set_xticks(range(visualizer.cols))
    ax1.set_yticks(range(visualizer.rows))
    ax1.set_xticklabels(range(visualizer.cols))
    ax1.set_yticklabels(range(visualizer.rows))
    
    # 그리드 라인 설정
    for i in range(visualizer.rows + 1):
        ax1.axhline(i - 0.5, color='gray', linewidth=0.5)
    for j in range(visualizer.cols + 1):
        ax1.axvline(j - 0.5, color='gray', linewidth=0.5)
    
    # 셀 레이블 초기화
    cell_texts = []
    for i in range(visualizer.rows):
        row_texts = []
        for j in range(visualizer.cols):
            text = ax1.text(j, i, visualizer.map_data[i][j], ha='center', va='center',
                           color='black', fontweight='bold')
            row_texts.append(text)
        cell_texts.append(row_texts)
    
    # 시간 표시 텍스트
    title = ax1.set_title("주차장 상태 (시간: 0:00)")
    
    # 범례 추가
    legend_elements = [
        Patch(facecolor=visualizer.CELL_COLORS['N'], label='경계/미사용'),
        Patch(facecolor=visualizer.CELL_COLORS['E'], label='입구/출구'),
        Patch(facecolor=visualizer.CELL_COLORS['R'], label='도로'),
        Patch(facecolor=visualizer.CELL_COLORS['P'], label='일반 주차면'),
        Patch(facecolor=visualizer.CELL_COLORS['C'], label='EV 충전소'),
        Patch(facecolor=visualizer.CELL_COLORS['O'], label='점유된 주차면'),
        Patch(facecolor=visualizer.CELL_COLORS['U'], label='사용 중인 충전소'),
        Patch(facecolor=visualizer.CELL_COLORS['M'], label='이동 중인 도로')
    ]
    ax1.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.05),
             ncol=4, frameon=True)
    
    def init():
        """애니메이션 초기화 함수"""
        return [img, title] + [text for row in cell_texts for text in row]
    
    def update(frame_idx):
        """
        애니메이션 프레임 업데이트 함수
        
        Args:
            frame_idx: 현재 프레임 인덱스
            
        Returns:
            업데이트된 아티스트 목록
        """
        if frame_idx >= len(frames):
            frame_idx = len(frames) - 1
            
        frame = frames[frame_idx]
        
        # 시간 표시 업데이트 (시:분 형식) - 시뮬레이션 시간으로 표시
        sim_time = frame['time']
        hours = int(sim_time / 3600)
        minutes = int((sim_time % 3600) / 60)
        seconds = int(sim_time % 60)
        title.set_text(f"주차장 상태 (시뮬레이션 시간: {hours:02d}:{minutes:02d}:{seconds:02d})")
        
        # 그리드 상태 업데이트
        visualizer.update_grid(frame['occupied'], frame['charging'])
        
        # 이동 중인 도로 셀 표시
        for r, c in frame['moving']:
            if 0 <= r < visualizer.rows and 0 <= c < visualizer.cols and visualizer.grid[r, c] == 'R':
                visualizer.grid[r, c] = 'M'
        
        # 셀 값 및 텍스트 업데이트
        cell_values = np.zeros(visualizer.grid.shape, dtype=int)
        for i, c in enumerate('NERPCOCUM'):
            cell_values[visualizer.grid == c] = i
        
        img.set_array(cell_values)
        
        for i in range(visualizer.rows):
            for j in range(visualizer.cols):
                cell_texts[i][j].set_text(visualizer.grid[i, j])
        
        return [img, title] + [text for row in cell_texts for text in row]
    
    # 애니메이션 생성
    anim = animation.FuncAnimation(
        fig, update, frames=len(frames), init_func=init, 
        interval=1000/fps, blit=False, cache_frame_data=False
    )
    
    # 애니메이션 저장
    print(f"[INFO] 애니메이션 저장 중... ({len(frames)} 프레임)")
    
    try:
        # ffmpeg가 설치되어 있으면 ffmpeg 사용
        anim.save(output_file, writer='ffmpeg', fps=fps, dpi=dpi, 
                 extra_args=['-vcodec', 'libx264', '-pix_fmt', 'yuv420p'])
        print(f"[INFO] 애니메이션이 {output_file}에 저장되었습니다.")
    except:
        # ffmpeg가 없으면 Pillow 사용
        print("[WARN] ffmpeg를 찾을 수 없어 Pillow로 저장합니다. 최상의 결과를 위해 ffmpeg를 설치하세요.")
        # .mp4 확장자를 .gif로 변경
        gif_output = output_file.replace('.mp4', '.gif')
        anim.save(gif_output, writer='pillow', fps=fps, dpi=dpi)
        print(f"[INFO] 애니메이션이 {gif_output}에 저장되었습니다.")
    
    plt.close()


def main():
    """
    메인 실행 함수
    """
    # 인자 파싱
    args = parse_arguments()
    
    # 로그 파일 확인
    if not os.path.exists(args.log_file):
        print(f"[ERROR] 로그 파일이 존재하지 않습니다: {args.log_file}")
        return 1
    
    # charge_log.csv가 입력된 경우 simulation_log.csv로 변경
    if "charge_log.csv" in args.log_file:
        simulation_log = args.log_file.replace("charge_log.csv", "simulation_log.csv")
        if os.path.exists(simulation_log):
            print(f"[INFO] charge_log.csv 대신 simulation_log.csv를 사용합니다.")
            args.log_file = simulation_log
        else:
            print(f"[ERROR] simulation_log.csv 파일을 찾을 수 없습니다: {simulation_log}")
            return 1
    
    print(f"[INFO] 애니메이션 생성 중... 로그 파일: {args.log_file}")
    
    # 애니메이션 데이터 준비
    frames = prepare_animation_data(args.log_file, args.speed)
    
    if not frames:
        print("[ERROR] 애니메이션 프레임을 생성할 수 없습니다.")
        return 1
    
    print(f"[INFO] 총 {len(frames)} 프레임의 애니메이션 데이터가 생성되었습니다.")
    
    # 출력 파일 경로를 상위 디렉토리로 변경
    output_file = os.path.join("..", args.output_file)
    
    # 애니메이션 생성 및 저장
    animate_parking(frames, output_file, args.fps, args.dpi)
    
    return 0


if __name__ == "__main__":
    sys.exit(main()) 