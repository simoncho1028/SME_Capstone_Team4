"""
주차장 상태를 시각화하는 모듈입니다.
"""
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib as mpl
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
import pandas as pd
import platform

from src.config import PARKING_MAP

# 한글 폰트 설정
if platform.system() == 'Windows':
    plt.rcParams['font.family'] = 'Malgun Gothic'  # 윈도우 한글 폰트
elif platform.system() == 'Darwin':  # macOS
    plt.rcParams['font.family'] = 'AppleGothic'    # 맥OS 한글 폰트
else:  # Linux
    plt.rcParams['font.family'] = 'NanumGothic'    # 리눅스 한글 폰트

mpl.rcParams['axes.unicode_minus'] = False   # 마이너스 기호 깨짐 방지

class ParkingVisualizer:
    """
    주차장 상태를 시각적으로 표현하는 클래스
    """
    
    # 셀 타입별 색상 정의
    CELL_COLORS = {
        'N': 'lightgray',   # 경계/미사용
        'E': 'gold',        # 입구/출구
        'R': 'white',       # 도로
        'P': 'lightgreen',  # 일반 주차면
        'C': 'skyblue',     # EV 충전소
        'O': 'tomato',      # 점유된 주차면 (일반)
        'U': 'royalblue',   # 사용 중인 충전소
        'M': 'plum'         # 이동 중인 도로 점유
    }
    
    def __init__(self, map_data: List[str] = PARKING_MAP):
        """
        시각화 도구를 초기화합니다.
        
        Args:
            map_data: 주차장 맵 데이터
        """
        self.map_data = map_data
        self.rows = len(map_data)
        self.cols = len(map_data[0])
        
        # 시각화를 위한 그리드 데이터 생성
        self.grid = np.array([[c for c in row] for row in map_data])
    
    def update_grid(self, occupied_spots: List[Tuple[int, int]], 
                    charging_spots: List[Tuple[int, int]],
                    moving_spots: Optional[List[Tuple[int, int]]] = None) -> None:
        """
        주차장 그리드 상태를 업데이트합니다.
        
        Args:
            occupied_spots: 점유된 일반 주차면 좌표 목록 [(r,c), ...]
            charging_spots: 사용 중인 충전소 좌표 목록 [(r,c), ...]
            moving_spots: 이동 중인 도로 셀 좌표 목록 [(r,c), ...]
        """
        # 그리드 초기화
        self.grid = np.array([[c for c in row] for row in self.map_data])
        
        # 점유된 일반 주차면 표시
        for r, c in occupied_spots:
            if 0 <= r < self.rows and 0 <= c < self.cols and self.grid[r, c] == 'P':
                self.grid[r, c] = 'O'
        
        # 사용 중인 충전소 표시
        for r, c in charging_spots:
            if 0 <= r < self.rows and 0 <= c < self.cols and self.grid[r, c] == 'C':
                self.grid[r, c] = 'U'
        
        # 이동 중인 도로 셀 표시
        if moving_spots:
            for r, c in moving_spots:
                if 0 <= r < self.rows and 0 <= c < self.cols and self.grid[r, c] == 'R':
                    self.grid[r, c] = 'M'
    
    def show(self, title: Optional[str] = None, figsize: Tuple[int, int] = (10, 8)):
        """
        주차장 상태를 시각화하여 표시합니다.
        
        Args:
            title: 시각화 제목
            figsize: 그래프 크기
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # 색상 매핑
        cmap = mcolors.ListedColormap([self.CELL_COLORS[c] for c in 'NERPCOCU'])
        bounds = range(len('NERPCOCU') + 1)
        norm = mcolors.BoundaryNorm(bounds, cmap.N)
        
        # 그리드 시각화
        cell_values = np.zeros(self.grid.shape, dtype=int)
        for i, c in enumerate('NERPCOCU'):
            cell_values[self.grid == c] = i
        
        ax.imshow(cell_values, cmap=cmap, norm=norm)
        
        # 셀 경계선 그리기
        for i in range(self.rows + 1):
            ax.axhline(i - 0.5, color='gray', linewidth=0.5)
        for j in range(self.cols + 1):
            ax.axvline(j - 0.5, color='gray', linewidth=0.5)
        
        # 셀 레이블 표시
        for i in range(self.rows):
            for j in range(self.cols):
                ax.text(j, i, self.grid[i, j], ha='center', va='center', 
                        color='black', fontweight='bold')
        
        # 좌표축 설정
        ax.set_xticks(range(self.cols))
        ax.set_yticks(range(self.rows))
        ax.set_xticklabels(range(self.cols))
        ax.set_yticklabels(range(self.rows))
        
        # 그리드 라인 설정
        ax.grid(True, color='gray', linestyle='-', linewidth=0.5)
        
        # 제목 설정
        if title:
            plt.title(title)
        
        # 범례 추가
        handles = [plt.Rectangle((0, 0), 1, 1, color=self.CELL_COLORS[c]) for c in 'NERPCOCU']
        labels = [
            '경계/미사용', 
            '입구/출구', 
            '도로', 
            '일반 주차면', 
            'EV 충전소',
            '점유된 주차면',
            '사용 중인 충전소'
        ]
        ax.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, -0.05),
                 ncol=4, frameon=True)
        
        plt.tight_layout()
        plt.show()
    
    def generate_animation_data(self, log_data: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        로그 데이터로부터 주차장 상태 변화를 애니메이션화할 데이터를 생성합니다.
        
        Args:
            log_data: 시뮬레이션 로그 데이터 (DataFrame)
            
        Returns:
            시간순으로 정렬된 주차장 상태 변화 데이터 목록
        """
        # 시간순으로 정렬
        df = log_data.sort_values('time')
        
        # 모든 시간에서의 주차장 상태 추적
        frames = []
        
        # 현재 점유된 주차면 및 충전소 추적
        occupied_spots = set()
        charging_spots = set()
        
        # 이전 시간
        prev_time = 0
        
        # 시간순으로 이벤트 처리
        for _, row in df.iterrows():
            time = row['time']
            event = row['event']
            pos = (row['pos_r'], row['pos_c'])
            
            # 30분마다 스냅샷 생성 (선택 사항)
            if time - prev_time >= 1800:  # 30분 = 1800초
                frames.append({
                    'time': prev_time,
                    'occupied': list(occupied_spots),
                    'charging': list(charging_spots)
                })
                prev_time = time
            
            # 이벤트에 따라 주차장 상태 업데이트
            if event == 'park_start':
                cell_type = self.map_data[pos[0]][pos[1]]
                if cell_type == 'P':
                    occupied_spots.add(pos)
                elif cell_type == 'C':
                    charging_spots.add(pos)
            
            elif event == 'charge_start':
                if pos in occupied_spots:
                    occupied_spots.remove(pos)
                charging_spots.add(pos)
            
            elif event == 'depart':
                if pos in occupied_spots:
                    occupied_spots.remove(pos)
                if pos in charging_spots:
                    charging_spots.remove(pos)
        
        # 마지막 상태 추가
        frames.append({
            'time': df['time'].max(),
            'occupied': list(occupied_spots),
            'charging': list(charging_spots)
        })
        
        return frames
    
    def save_state_image(self, occupied_spots: List[Tuple[int, int]], 
                        charging_spots: List[Tuple[int, int]], 
                        filename: str, title: Optional[str] = None) -> None:
        """
        주차장 상태를 이미지 파일로 저장합니다.
        
        Args:
            occupied_spots: 점유된 일반 주차면 좌표 목록
            charging_spots: 사용 중인 충전소 좌표 목록
            filename: 저장할 파일 이름
            title: 이미지 제목
        """
        self.update_grid(occupied_spots, charging_spots)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # 색상 매핑
        cmap = mcolors.ListedColormap([self.CELL_COLORS[c] for c in 'NERPCOCU'])
        bounds = range(len('NERPCOCU') + 1)
        norm = mcolors.BoundaryNorm(bounds, cmap.N)
        
        # 그리드 시각화
        cell_values = np.zeros(self.grid.shape, dtype=int)
        for i, c in enumerate('NERPCOCU'):
            cell_values[self.grid == c] = i
        
        ax.imshow(cell_values, cmap=cmap, norm=norm)
        
        # 셀 경계선 그리기
        for i in range(self.rows + 1):
            ax.axhline(i - 0.5, color='gray', linewidth=0.5)
        for j in range(self.cols + 1):
            ax.axvline(j - 0.5, color='gray', linewidth=0.5)
        
        # 셀 레이블 표시
        for i in range(self.rows):
            for j in range(self.cols):
                ax.text(j, i, self.grid[i, j], ha='center', va='center', 
                        color='black', fontweight='bold')
        
        # 좌표축 설정
        ax.set_xticks(range(self.cols))
        ax.set_yticks(range(self.rows))
        ax.set_xticklabels(range(self.cols))
        ax.set_yticklabels(range(self.rows))
        
        # 그리드 라인 설정
        ax.grid(True, color='gray', linestyle='-', linewidth=0.5)
        
        # 제목 설정
        if title:
            plt.title(title)
        
        # 범례 추가
        handles = [plt.Rectangle((0, 0), 1, 1, color=self.CELL_COLORS[c]) for c in 'NERPCOCU']
        labels = [
            '경계/미사용', 
            '입구/출구', 
            '도로', 
            '일반 주차면', 
            'EV 충전소',
            '점유된 주차면',
            '사용 중인 충전소'
        ]
        ax.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, -0.05),
                 ncol=4, frameon=True)
        
        plt.tight_layout()
        plt.savefig(filename)
        plt.close() 