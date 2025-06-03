#!/usr/bin/env python3
"""
4개 층 주차장 시뮬레이션 애니메이션 (simulation_log.csv 기반)

- 2x2 subplot에 4개 층 동시 표시
- 주차면/충전소 색상 명확 구분
- park_success/charge_start/depart 이벤트에 따라 상태 변화
- 각 층별/전체 통계, 범례, 격자, 제목 표시
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
import platform
from datetime import datetime
from typing import List, Dict, Any

from src.config import PARKING_MAPS

# 한글 폰트 설정
if platform.system() == 'Windows':
    plt.rcParams['font.family'] = 'Malgun Gothic'
elif platform.system() == 'Darwin':
    plt.rcParams['font.family'] = 'AppleGothic'
else:
    plt.rcParams['font.family'] = 'NanumGothic'
plt.rcParams['axes.unicode_minus'] = False

# 색상 매핑 (0~6)
COLORS = [
    '#BDBDBD',  # 0: 경계
    '#FFA500',  # 1: 입구
    '#FFFFFF',  # 2: 도로
    '#E3F2FD',  # 3: 빈 주차
    '#1976D2',  # 4: 점유 주차
    '#E8F5E8',  # 5: 빈 충전
    '#388E3C',  # 6: 점유 충전
]

FLOORS = ['GF', 'B1F', 'B2F', 'B3F']
FLOOR_TITLES = {'GF': '지상층', 'B1F': '지하1층', 'B2F': '지하2층', 'B3F': '지하3층'}

# 셀 상태 변환 함수
def get_cell_value(cell, occupied_parking, occupied_charging, r, c):
    if cell == 'N':
        return 0
    elif cell == 'E':
        return 1
    elif cell == 'R':
        return 2
    elif cell == 'P':
        return 4 if (r, c) in occupied_parking else 3
    elif cell == 'C':
        return 6 if (r, c) in occupied_charging else 5
    return 0

def prepare_frames_from_log(df: pd.DataFrame, frame_interval: float = 30.0) -> List[Dict[str, Any]]:
    """simulation_log.csv를 읽어 프레임별 상태를 생성"""
    df = df.sort_values(by='time').reset_index(drop=True)
    start_time = df['time'].min() if not df.empty else 0
    end_time = df['time'].max() if not df.empty else 0
    frames = []
    event_idx = 0
    current_time = start_time
    vehicle_states = {}  # id -> {'floor', 'pos', 'state'}
    occupied_spots = {}  # (floor, r, c) -> 'parked'/'charging'
    while current_time <= end_time:
        # 이벤트 처리
        while event_idx < len(df) and df.loc[event_idx, 'time'] <= current_time:
            event = df.loc[event_idx]
            vid = event['id']
            etype = event['event']
            floor = event['floor']
            try:
                pos = (int(event['pos_r']), int(event['pos_c']))
            except:
                event_idx += 1
                continue
            if etype == 'park_success':
                # 기존 위치 제거
                if vid in vehicle_states:
                    old = vehicle_states[vid]
                    old_key = (old['floor'], old['pos'][0], old['pos'][1])
                    if old_key in occupied_spots:
                        del occupied_spots[old_key]
                vehicle_states[vid] = {'floor': floor, 'pos': pos, 'state': 'parked'}
                occupied_spots[(floor, pos[0], pos[1])] = 'parked'
            elif etype == 'charge_start':
                if vid in vehicle_states:
                    old = vehicle_states[vid]
                    old_key = (old['floor'], old['pos'][0], old['pos'][1])
                    if old_key in occupied_spots:
                        del occupied_spots[old_key]
                vehicle_states[vid] = {'floor': floor, 'pos': pos, 'state': 'charging'}
                occupied_spots[(floor, pos[0], pos[1])] = 'charging'
            elif etype == 'depart':
                if vid in vehicle_states:
                    old = vehicle_states[vid]
                    old_key = (old['floor'], old['pos'][0], old['pos'][1])
                    if old_key in occupied_spots:
                        del occupied_spots[old_key]
                    del vehicle_states[vid]
            event_idx += 1
        # 층별 점유 set
        floor_states = {}
        for floor in FLOORS:
            parking, charging = set(), set()
            total_p, total_c = 0, 0
            for r, row in enumerate(PARKING_MAPS[floor]):
                for c, cell in enumerate(row):
                    if cell == 'P':
                        total_p += 1
                    elif cell == 'C':
                        total_c += 1
            for (f, r, c), state in occupied_spots.items():
                if f == floor:
                    if state == 'parked':
                        parking.add((r, c))
                    elif state == 'charging':
                        charging.add((r, c))
            floor_states[floor] = {
                'occupied_parking': parking,
                'occupied_charging': charging,
                'total_parking': total_p,
                'total_charging': total_c
            }
        frames.append({
            'time': current_time,
            'floors': floor_states,
            'total_vehicles': len(vehicle_states)
        })
        # 다음 프레임
        if current_time >= end_time:
            break
        current_time += frame_interval
        if event_idx < len(df) and current_time > df.loc[event_idx, 'time']:
            current_time = df.loc[event_idx, 'time']
    return frames

class ParkingAnimationVisualizer:
    def __init__(self, floors, parking_maps, colors):
        self.floors = floors
        self.maps = parking_maps
        self.colors = colors
        self.cmap = mcolors.ListedColormap(colors)
        self.norm = mcolors.BoundaryNorm(np.arange(len(colors)+1), len(colors))
        self.floor_positions = {'GF': (0,0), 'B1F': (0,1), 'B2F': (1,0), 'B3F': (1,1)}
    def animate(self, frames, output_file, fps=10, dpi=100):
        fig, axes = plt.subplots(2,2,figsize=(20,16))
        fig.suptitle('아파트 주차장 실시간 현황', fontsize=20, fontweight='bold')
        floor_images, floor_stats = {}, {}
        for floor in self.floors:
            row, col = self.floor_positions[floor]
            ax = axes[row, col]
            grid = np.array([list(r) for r in self.maps[floor]])
            rows, cols = grid.shape
            grid_values = np.zeros((rows, cols), dtype=int)
            img = ax.imshow(grid_values, cmap=self.cmap, norm=self.norm, aspect='equal')
            floor_images[floor] = img
            ax.set_title(f'{floor} ({FLOOR_TITLES[floor]})', fontsize=14, fontweight='bold')
            stats = ax.text(0.5, -0.1, '', transform=ax.transAxes, ha='center', va='top', fontsize=10,
                            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
            floor_stats[floor] = stats
            ax.set_xticks([]); ax.set_yticks([])
            for i in range(rows+1):
                ax.axhline(i-0.5, color='gray', linewidth=0.3, alpha=0.7)
            for j in range(cols+1):
                ax.axvline(j-0.5, color='gray', linewidth=0.3, alpha=0.7)
        legend_elements = [
            Patch(facecolor=self.colors[3], label='빈 주차면'),
            Patch(facecolor=self.colors[4], label='점유 주차면'),
            Patch(facecolor=self.colors[5], label='빈 충전소'),
            Patch(facecolor=self.colors[6], label='점유 충전소'),
            Patch(facecolor=self.colors[2], label='도로'),
            Patch(facecolor=self.colors[1], label='입구/출구'),
            Patch(facecolor=self.colors[0], label='경계/미사용'),
        ]
        fig.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, 0.02), ncol=4, frameon=True, fontsize=12)
        overall_stats = fig.text(0.5, 0.95, '', ha='center', va='top', fontsize=14, fontweight='bold', transform=fig.transFigure)
        def update(idx):
            frame = frames[idx if idx < len(frames) else -1]
            sim_time = frame['time']
            hours = int(sim_time // 3600)
            minutes = int((sim_time % 3600) // 60)
            overall_stats.set_text(f'시뮬레이션 시간: {hours:02d}:{minutes:02d} | 총 주차 차량: {frame["total_vehicles"]}대')
            for floor in self.floors:
                grid = np.array([list(r) for r in self.maps[floor]])
                rows, cols = grid.shape
                grid_values = np.zeros((rows, cols), dtype=int)
                state = frame['floors'][floor]
                for i in range(rows):
                    for j in range(cols):
                        cell = grid[i, j]
                        grid_values[i, j] = get_cell_value(cell, state['occupied_parking'], state['occupied_charging'], i, j)
                floor_images[floor].set_array(grid_values)
                p_total = state['total_parking']; p_occ = len(state['occupied_parking'])
                c_total = state['total_charging']; c_occ = len(state['occupied_charging'])
                p_rate = (p_occ/p_total*100) if p_total else 0
                c_rate = (c_occ/c_total*100) if c_total else 0
                stats_str = f'주차: {p_occ}/{p_total} ({p_rate:.1f}%)\n충전: {c_occ}/{c_total} ({c_rate:.1f}%)'
                floor_stats[floor].set_text(stats_str)
            return list(floor_images.values()) + [overall_stats] + list(floor_stats.values())
        anim = animation.FuncAnimation(fig, update, frames=len(frames), interval=1000/fps, blit=False, cache_frame_data=False)
        print(f"[INFO] 애니메이션 저장 중... ({len(frames)} 프레임)")
        try:
            anim.save(output_file, writer='ffmpeg', fps=fps, dpi=dpi, extra_args=['-vcodec', 'libx264', '-pix_fmt', 'yuv420p'])
            print(f"[INFO] 애니메이션이 {output_file}에 저장되었습니다.")
        except Exception as e:
            print(f"[WARN] ffmpeg 저장 실패: {e}")
            gif_output = output_file.replace('.mp4', '.gif')
            anim.save(gif_output, writer='pillow', fps=fps, dpi=dpi)
            print(f"[INFO] 애니메이션이 {gif_output}에 저장되었습니다.")
        plt.close()

def parse_arguments():
    parser = argparse.ArgumentParser(description="4개 층 주차장 애니메이션", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("log_file", type=str, help="simulation_log.csv 경로")
    parser.add_argument("output_file", type=str, nargs='?', default="parking_animation.mp4", help="저장할 mp4 파일명")
    parser.add_argument("--fps", type=int, default=10, help="프레임 속도(FPS)")
    parser.add_argument("--dpi", type=int, default=100, help="이미지 해상도(DPI)")
    parser.add_argument("--interval", type=float, default=30.0, help="프레임 간격(초)")
    return parser.parse_args()

def main():
    args = parse_arguments()
    if not os.path.exists(args.log_file):
        print(f"[ERROR] 로그 파일을 찾을 수 없습니다: {args.log_file}")
        return 1
    df = pd.read_csv(args.log_file)
    print("[INFO] 프레임 데이터 생성 중...")
    frames = prepare_frames_from_log(df, frame_interval=args.interval)
    if not frames:
        print("[ERROR] 프레임 데이터가 없습니다.")
        return 1
    visualizer = ParkingAnimationVisualizer(FLOORS, PARKING_MAPS, COLORS)
    print(f"[INFO] 애니메이션 생성 및 저장 중... (FPS: {args.fps}, DPI: {args.dpi})")
    visualizer.animate(frames, args.output_file, fps=args.fps, dpi=args.dpi)
    print(f"[INFO] 애니메이션이 저장되었습니다: {args.output_file}")
    return 0

if __name__ == "__main__":
    sys.exit(main()) 