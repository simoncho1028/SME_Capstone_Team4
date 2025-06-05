#!/usr/bin/env python3
"""
주차장 애니메이션 시각화 도구
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import platform
import pandas as pd
from matplotlib.animation import FuncAnimation
import os
from src.utils.parking_map_loader import ParkingMapLoader
import matplotlib
import matplotlib.patches as mpatches
matplotlib.use('Agg')  # GUI 없이 렌더링

# 한글 폰트 설정
if platform.system() == 'Windows':
    plt.rcParams['font.family'] = 'Malgun Gothic'
elif platform.system() == 'Darwin':
    plt.rcParams['font.family'] = 'AppleGothic'
else:
    plt.rcParams['font.family'] = 'NanumGothic'

class ParkingAnimator:
    # 색상 매핑
    COLORS = {
        'R': '#D3D3D3',    # 길: 밝은 회색
        'P': '#90EE90',    # 주차면: 연한 초록색
        'E': '#FFD700',    # 입구: 금색
        'X': '#FFA500',    # 출구: 주황색
        'B1': '#ADD8E6',   # 아파트: 연한 파란색
        'O': '#FF6B6B',    # 주차장 점유중: 빨간색
        'C': '#4CAF50',    # EV 충전소: 진한 초록색
        'CO': '#FF4500'    # EV 충전소 점유중: 진한 주황색
    }
    
    # 범례 레이블
    LEGEND_LABELS = {
        'R': '도로',
        'P': '주차면',
        'E': '입구',
        'X': '출구',
        'O': '주차중',
        'C': 'EV 충전소',
        'CO': '충전중'
    }

    def __init__(self, json_dir: str = "json", charger_positions=None):
        """애니메이터 초기화"""
        self.map_loader = ParkingMapLoader(json_dir)
        self.layouts = self.map_loader.load_all_maps()
        
        # 층 이름 매핑
        self.floor_mapping = {
            'Ground': 'GF',
            'B1': 'B1F',
            'B2': 'B2F',
            'B3': 'B3F'
        }
        self.reverse_floor_mapping = {v: k for k, v in self.floor_mapping.items()}
        
        # 충전소 위치 저장
        self.charger_positions = charger_positions or {}
        
        # 그림 설정
        plt.ioff()  # 인터랙티브 모드 끄기
        self.fig = plt.figure(figsize=(15, 17))  # 범례를 위한 공간 추가
        
        # 서브플롯 그리드 설정 (3x2 그리드, 위쪽 2x2는 주차장, 아래쪽은 범례)
        gs = self.fig.add_gridspec(3, 2, height_ratios=[4, 4, 1])
        self.axes = [
            [self.fig.add_subplot(gs[0, 0]), self.fig.add_subplot(gs[0, 1])],
            [self.fig.add_subplot(gs[1, 0]), self.fig.add_subplot(gs[1, 1])]
        ]
        
        self.fig.suptitle('주차장 시뮬레이션', fontsize=16, y=0.95)
        
        # 각 층의 Rectangle 객체를 저장할 딕셔너리
        self.rectangles = {floor: {} for floor in ['GF', 'B1F', 'B2F', 'B3F']}
        
        # 주차 상태를 저장할 딕셔너리
        self.parking_state = {}  # (floor, pos) -> status
        
        # 초기 레이아웃 그리기
        self._setup_initial_layout()
        
        # 범례 추가
        self._add_legend(gs[2, :])
        
        # 창 크기 조정
        plt.tight_layout()

    def _add_legend(self, subplot_spec):
        """범례 추가"""
        ax_legend = self.fig.add_subplot(subplot_spec)
        ax_legend.axis('off')
        
        # 범례 항목 생성
        legend_patches = []
        for key, label in self.LEGEND_LABELS.items():
            color = self.COLORS[key]
            patch = mpatches.Patch(facecolor=color, edgecolor='black', label=label)
            legend_patches.append(patch)
        
        # 범례 배치 (2줄로)
        ax_legend.legend(handles=legend_patches, loc='center', ncol=4, 
                        fontsize=12, frameon=True, edgecolor='black')
        
    def _setup_initial_layout(self):
        """초기 레이아웃 설정"""
        floor_titles = {
            'GF': '지상층',
            'B1F': '지하 1층',
            'B2F': '지하 2층',
            'B3F': '지하 3층'
        }
        
        for idx, (floor, layout) in enumerate(self.layouts.items()):
            ax = self.axes[idx // 2][idx % 2]
            ax.set_title(floor_titles[floor], fontsize=14)
            
            height = len(layout)
            width = len(layout[0])
            
            # 각 셀 그리기
            for i in range(height):
                for j in range(width):
                    cell = layout[i][j]
                    # 충전소 위치 확인
                    is_charger = False
                    if self.charger_positions:
                        sim_floor = self.reverse_floor_mapping[floor]
                        if sim_floor in self.charger_positions and (i, j) in self.charger_positions[sim_floor]:
                            is_charger = True
                    
                    # 색상 결정
                    if is_charger:
                        color = self.COLORS['C']
                    else:
                        color = self.COLORS.get(cell, 'white')
                    
                    rect = plt.Rectangle((j, height-i-1), 1, 1, 
                                      facecolor=color, 
                                      edgecolor='black', 
                                      linewidth=0.5)
                    ax.add_patch(rect)
                    # 주차 가능한 공간의 Rectangle 객체 저장
                    if cell == 'P' or is_charger:
                        self.rectangles[floor][(i, j)] = rect
                        self.parking_state[(floor, (i, j))] = 'empty'
            
            ax.set_xlim(-0.2, width+0.2)
            ax.set_ylim(-0.2, height+0.2)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.grid(True, color='black', linewidth=0.5)

    def update_spot(self, floor: str, position: tuple, status: str):
        """특정 주차 공간의 상태 업데이트"""
        # 층 이름 변환
        internal_floor = self.floor_mapping.get(floor, floor)
        
        if position in self.rectangles[internal_floor]:
            # 충전소 여부 확인
            sim_floor = floor
            is_charger = (sim_floor in self.charger_positions and position in self.charger_positions[sim_floor])
            
            # 상태에 따른 색상 결정
            if status == 'empty':
                color = self.COLORS['C'] if is_charger else self.COLORS['P']
            elif status == 'charging':
                color = self.COLORS['CO']
            else:  # occupied
                color = self.COLORS['O']
            
            self.rectangles[internal_floor][position].set_facecolor(color)
            self.parking_state[(internal_floor, position)] = status

    def animate(self, log_file: str, speed: float = 0.1, save_path: str = None):
        """주차장 상태 애니메이션 실행 또는 저장"""
        # CSV 파일 읽기
        df = pd.read_csv(log_file)
        
        # 관련 이벤트만 필터링
        events_df = df[df['event'].isin(['park_success', 'depart', 'charge_start'])].copy()
        events_df = events_df.sort_values('time')  # 시간순 정렬
        
        if save_path:  # MP4로 저장
            # 시간 간격으로 이벤트 그룹화
            time_interval = 300  # 5분 간격
            max_time = events_df['time'].max()
            frames = []
            
            # 시간 간격별로 이벤트 처리
            for start_time in range(0, int(max_time) + time_interval, time_interval):
                end_time = start_time + time_interval
                interval_events = events_df[
                    (events_df['time'] >= start_time) & 
                    (events_df['time'] < end_time)
                ]
                
                if len(interval_events) > 0:
                    frame_events = []
                    for _, row in interval_events.iterrows():
                        if pd.isna(row['pos_r']) or pd.isna(row['pos_c']) or pd.isna(row['floor']):
                            continue
                        
                        frame_events.append({
                            'floor': row['floor'],
                            'pos': (int(row['pos_r']), int(row['pos_c'])),
                            'status': 'occupied' if row['event'] == 'park_success' else \
                                     'charging' if row['event'] == 'charge_start' else \
                                     'empty',
                            'time': end_time
                        })
                    
                    if frame_events:
                        frames.append(frame_events)

            def update(frame_events):
                for event in frame_events:
                    self.update_spot(event['floor'], event['pos'], event['status'])
                self.fig.suptitle(f'주차장 시뮬레이션 - {frame_events[0]["time"]/3600:.1f}시간', fontsize=16, y=0.95)
                
                # 모든 axes의 모든 패치 반환
                artists = []
                for axes_row in self.axes:
                    for ax in axes_row:
                        artists.extend(ax.get_children())
                return artists

            # 애니메이션 생성 및 저장
            print(f"\n애니메이션 생성 중... (총 {len(frames)}개 프레임)")
            anim = FuncAnimation(
                self.fig,
                update,
                frames=frames,
                interval=max(100, speed * 1000),  # 최소 100ms
                blit=True,
                repeat=False  # 반복 재생 비활성화
            )
            
            # MP4 저장 설정
            writer = 'ffmpeg'
            fps = min(20, max(5, int(1/speed)))  # 최소 5fps, 최대 20fps
            
            # 저장 품질 설정
            writer_kwargs = {
                'fps': fps,
                'bitrate': 2000,
                'codec': 'h264',
                'extra_args': ['-pix_fmt', 'yuv420p', '-preset', 'fast']  # 인코딩 속도 개선
            }
            
            # 저장
            print(f"애니메이션 저장 중... (FPS: {fps})")
            anim.save(
                save_path,
                writer=writer,
                dpi=100,
                **writer_kwargs
            )
            print(f"애니메이션이 저장되었습니다: {save_path}")
            plt.close()
            
        else:  # 실시간 애니메이션
            plt.ion()  # 인터랙티브 모드 켜기
            
            for _, row in events_df.iterrows():
                if pd.isna(row['pos_r']) or pd.isna(row['pos_c']) or pd.isna(row['floor']):
                    continue
                
                floor = row['floor']
                pos = (int(row['pos_r']), int(row['pos_c']))
                event = row['event']
                
                # 이벤트에 따른 상태 결정
                if event == 'park_success':
                    status = 'occupied'
                elif event == 'charge_start':
                    status = 'charging'
                else:  # depart
                    status = 'empty'
                
                self.fig.suptitle(f'주차장 시뮬레이션 - {row["time"]/3600:.1f}시간', fontsize=16, y=0.95)
                self.update_spot(floor, pos, status)
                plt.pause(speed)
            
            plt.ioff()
            plt.show()

def run_animation(json_dir: str, log_file: str, speed: float = 0.1, save_video: bool = False, charger_positions=None):
    """애니메이션 실행"""
    animator = ParkingAnimator(json_dir, charger_positions)
    
    if save_video:
        # log_file과 같은 디렉토리에 저장
        output_dir = os.path.dirname(log_file)
        video_path = os.path.join(output_dir, "parking_animation.mp4")
        animator.animate(log_file, speed, save_path=video_path)
    else:
        animator.animate(log_file, speed) 