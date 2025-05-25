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
from datetime import datetime

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
        default=80,
        help="이미지 해상도 (DPI)"
    )
    
    parser.add_argument(
        "--speed", 
        type=float, 
        default=60.0,
        help="시뮬레이션 속도 (실제 1초당 시뮬레이션 시간 초)"
    )
    
    return parser.parse_args()


def prepare_animation_data(df: pd.DataFrame, speed_factor: float = 60.0) -> List[Dict[str, Any]]:
    """
    시뮬레이션 로그 데이터를 애니메이션 프레임으로 변환합니다.
    
    Args:
        df: 시뮬레이션 로그 데이터프레임
        speed_factor: 시뮬레이션 속도 (실제 1초당 시뮬레이션 시간 초)
        
    Returns:
        애니메이션 프레임 리스트
    """
    # 컬럼명 확인 및 매핑
    column_mapping = {
        'pos_r': 'row',
        'pos_c': 'col'
    }
    
    # 컬럼명 변경
    df = df.rename(columns=column_mapping)
    
    # 필요한 컬럼이 있는지 확인
    required_columns = ['time', 'row', 'col', 'event', 'id']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        print(f"[ERROR] CSV 파일에 필요한 컬럼이 없습니다: {missing_columns}")
        print(f"[INFO] 현재 CSV 파일의 컬럼: {list(df.columns)}")
        return []
    
    # 시간 기준으로 정렬
    df = df.sort_values(by='time').reset_index(drop=True)
    
    frames = []
    
    # 시간 간격 설정 (프레임 수 조절)
    frame_interval = speed_factor / 5  # 5로 나누어 프레임 수 감소
    
    # 시간 범위 설정
    start_time = df['time'].min() if not df.empty else 0
    end_time = df['time'].max() if not df.empty else 0
    
    # 차량별 최신 상태 및 위치 추적
    vehicle_latest_state = {}
    vehicle_latest_pos = {}
    
    # 프레임 생성
    current_time = start_time
    event_idx = 0 # 현재 처리할 이벤트 인덱스
    
    while current_time <= end_time or (current_time == start_time and end_time == 0):
        
        # 현재 시간까지 발생한 이벤트 처리하여 차량 상태 업데이트
        while event_idx < len(df) and df.loc[event_idx, 'time'] <= current_time:
            event = df.loc[event_idx]
            vehicle_id = event['id']
            pos = (event['row'], event['col'])
            event_type = event['event']
            
            vehicle_latest_pos[vehicle_id] = pos
            vehicle_latest_state[vehicle_id] = event_type
            
            event_idx += 1
            
        # 현재 프레임의 주차장 상태 결정
        occupied = set()
        charging = set()
        moving = set()
        
        for v_id, state in vehicle_latest_state.items():
            pos = vehicle_latest_pos[v_id]
            map_cell_type = PARKING_MAP[pos[0]][pos[1]]
            
            if state == 'park_start':
                occupied.add(pos)
            elif state == 'charge_start':
                occupied.add(pos)
                charging.add(pos)
            elif state == 'move':
                 # 도로 셀인 경우에만 이동 중으로 표시
                if map_cell_type == 'R':
                    moving.add(pos)
            # 'depart' 또는 'charge_end' 상태인 차량은 표시하지 않음 (이미 나갔거나 상태 변경)
            
        # 프레임 추가
        frames.append({
            'time': current_time,
            'occupied': occupied,
            'charging': charging,
            'moving': moving
        })
        
        # 다음 프레임 시간으로 이동
        # 마지막 이벤트 시간을 초과하지 않도록 보정
        if current_time == end_time and end_time > 0:
             break # 마지막 이벤트 시간에 도달했으면 종료
             
        current_time += frame_interval
        # 다음 current_time이 다음 이벤트 시간보다 뒤쳐지지 않도록 보정
        if event_idx < len(df) and current_time < df.loc[event_idx, 'time']:
             current_time = df.loc[event_idx, 'time']
        
    # 마지막 상태를 반영하는 프레임 추가 (선택 사항)
    if not frames or (frames[-1]['time'] < end_time and end_time > 0):
         # 마지막 이벤트까지 처리
        while event_idx < len(df):
             event = df.loc[event_idx]
             vehicle_id = event['id']
             pos = (event['row'], event['col'])
             event_type = event['event']
            
             vehicle_latest_pos[vehicle_id] = pos
             vehicle_latest_state[vehicle_id] = event_type
             event_idx += 1
             
        occupied = set()
        charging = set()
        moving = set()
        
        for v_id, state in vehicle_latest_state.items():
            pos = vehicle_latest_pos[v_id]
            map_cell_type = PARKING_MAP[pos[0]][pos[1]]
            
            if state == 'park_start':
                occupied.add(pos)
            elif state == 'charge_start':
                occupied.add(pos)
                charging.add(pos)
            elif state == 'move':
                 if map_cell_type == 'R':
                     moving.add(pos)
                     
        frames.append({
             'time': end_time,
             'occupied': occupied,
             'charging': charging,
             'moving': moving
        })
        
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
        
        # 시간 표시 업데이트 (시:분 형식)
        sim_time = frame['time']
        hours = int(sim_time / 3600)
        minutes = int((sim_time % 3600) / 60)
        seconds = int(sim_time % 60)
        title.set_text(f"주차장 상태 (시뮬레이션 시간: {hours:02d}:{minutes:02d}:{seconds:02d})")
        
        # 그리드 상태 업데이트
        visualizer.update_grid(frame['occupied'], frame['charging'])
        
        # 이동 중인 도로 셀 표시
        for r, c in frame.get('moving', set()):
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
    
    # 로그 파일 경로 처리
    log_file = args.log_file
    if not os.path.exists(log_file):
        print(f"[ERROR] 로그 파일을 찾을 수 없습니다: {log_file}")
        return 1
    
    # 출력 파일 경로 설정
    if args.output_file:
        output_path = args.output_file
    else:
        # 로그 파일이 있는 디렉토리에서 기본 파일명 생성
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(os.path.dirname(log_file), f"parking_animation_{timestamp}.mp4")
    
    print(f"[INFO] 로그 파일 읽는 중: {log_file}")
    try:
        # CSV 파일 읽기
        df = pd.read_csv(log_file)
        
        # 애니메이션 데이터 준비
        print("[INFO] 애니메이션 데이터 준비 중...")
        frames = prepare_animation_data(df, args.speed)
        
        if frames:
            # 애니메이션 생성 및 저장
            print(f"[INFO] 애니메이션 생성 중... (FPS: {args.fps}, DPI: {args.dpi})")
            animate_parking(frames, output_path, args.fps, args.dpi)
            print(f"[INFO] 애니메이션이 저장되었습니다: {output_path}")
        else:
            print("[ERROR] 애니메이션 프레임을 생성할 수 없습니다.")
            return 1
            
    except Exception as e:
        print(f"[ERROR] 파일 처리 중 오류가 발생했습니다: {str(e)}")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main()) 