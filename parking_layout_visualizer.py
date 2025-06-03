#!/usr/bin/env python3
"""
주차장 레이아웃 시각화 도구

각 층별 주차장 공간을 그리드 형태로 시각화하고 이미지 파일로 저장합니다.
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Dict, List
import platform
import matplotlib.patches as mpatches

# 한글 폰트 설정
if platform.system() == 'Windows':
    plt.rcParams['font.family'] = 'Malgun Gothic'
elif platform.system() == 'Darwin':
    plt.rcParams['font.family'] = 'AppleGothic'
else:
    plt.rcParams['font.family'] = 'NanumGothic'

from src.utils.parking_map_loader import ParkingMapLoader

class ParkingLayoutVisualizer:
    # 색상 매핑
    COLORS = {
        'R': '#D3D3D3',    # 길: 밝은 회색
        'P': '#90EE90',    # 주차면: 연한 초록색
        'E': '#FFD700',    # 입구: 금색
        'X': '#FFA500',    # 출구: 주황색
        'B1': '#ADD8E6',   # 1동: 연한 파란색
        'B2': '#ADD8E6',   # 2동
        'B3': '#ADD8E6',   # 3동
        'B4': '#ADD8E6',   # 4동
        'B5': '#ADD8E6',   # 5동
        'B6': '#ADD8E6',   # 6동
        'B7': '#ADD8E6',   # 7동
        'B8': '#ADD8E6',   # 8동
        'O': '#FF6B6B',    # 주차장 점유중: 빨간색
        'C': '#4CAF50',    # EV 충전소: 진한 초록색
        'CO': '#FF4500'    # EV 충전소 점유중: 진한 주황색
    }

    # 범례 텍스트
    LEGEND_LABELS = {
        'R': '도로',
        'P': '주차면',
        'E': '입구',
        'X': '출구',
        'B1': '아파트',
        'O': '주차장 점유중',
        'C': 'EV 충전소',
        'CO': 'EV 충전소 점유중'
    }

    def __init__(self, output_dir: str = "layout_images"):
        """
        주차장 레이아웃 시각화 도구 초기화
        
        Args:
            output_dir: 이미지 저장 디렉토리
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.map_loader = ParkingMapLoader()
        
        # DPI 설정
        self.dpi = 100
        
        # 그리드 크기 설정
        self.cell_size = 1.2  # 셀 크기 증가
        self.font_size = 20   # 폰트 크기 증가

    def visualize_all_floors(self):
        """모든 층의 주차장 레이아웃을 시각화하고 저장합니다."""
        # 모든 층의 맵 데이터 로드
        floor_maps = self.map_loader.load_all_maps()
        
        # 각 층별로 시각화
        for floor_name, layout in floor_maps.items():
            self._visualize_floor(floor_name, layout)
            
    def _visualize_floor(self, floor_name: str, layout: List[List[str]]):
        """
        특정 층의 주차장 레이아웃을 시각화하고 저장합니다.
        
        Args:
            floor_name: 층 이름
            layout: 레이아웃 데이터
        """
        # 레이아웃 크기 계산
        height = len(layout)
        width = len(layout[0])
        
        # 그림 크기 설정 (적절한 종횡비 유지)
        fig_width = width * self.cell_size
        fig_height = height * self.cell_size + 2  # 범례를 위한 추가 공간
        
        # 새 그림 생성
        fig = plt.figure(figsize=(fig_width, fig_height), dpi=self.dpi)
        
        # 메인 플롯을 위한 서브플롯 생성 (여백 줄임)
        ax_main = plt.subplot2grid((7, 1), (0, 0), rowspan=6)
        plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.15)
        
        # 격자 그리기
        for i in range(height):
            for j in range(width):
                cell = layout[i][j]
                # 건물(B1~B8)은 모두 같은 색상 사용
                if cell.startswith('B'):
                    color = self.COLORS['B1']
                else:
                    color = self.COLORS.get(cell, 'white')
                
                # 셀 그리기
                rect = plt.Rectangle((j, height-i-1), 1, 1, facecolor=color, edgecolor='black', linewidth=0.5)
                ax_main.add_patch(rect)
        
        # 그리드 선 추가
        ax_main.grid(True, color='black', linewidth=0.5)
        ax_main.set_aspect('equal')
        
        # 축 범위 설정
        ax_main.set_xlim(-0.2, width+0.2)
        ax_main.set_ylim(-0.2, height+0.2)
        
        # 축 눈금 제거
        ax_main.set_xticks([])
        ax_main.set_yticks([])
        
        # 제목 설정
        floor_titles = {
            'GF': '지상층',
            'B1F': '지하 1층',
            'B2F': '지하 2층',
            'B3F': '지하 3층'
        }
        ax_main.set_title(f'주차장 레이아웃 - {floor_titles.get(floor_name, floor_name)}', 
                         pad=20, fontsize=self.font_size+10)
        
        # 범례 추가 (크기 증가 및 2줄로 배치)
        legend_patches = []
        for key, label in self.LEGEND_LABELS.items():
            color = self.COLORS[key]
            patch = mpatches.Patch(facecolor=color, label=label, edgecolor='black', linewidth=1)
            legend_patches.append(patch)
        
        # 범례를 위한 새로운 축 생성
        ax_legend = plt.subplot2grid((7, 1), (6, 0))
        ax_legend.axis('off')
        
        # 범례를 2줄로 배치하고 크기 증가
        legend = ax_legend.legend(handles=legend_patches, loc='center', 
                                ncol=4, bbox_to_anchor=(0.5, 0.5),
                                fontsize=self.font_size, 
                                borderaxespad=0,
                                handlelength=3,
                                handleheight=2)
        
        # 범례 테두리 추가
        legend.get_frame().set_linewidth(1)
        legend.get_frame().set_edgecolor('black')
        
        # 이미지 저장
        output_path = self.output_dir / f'parking_layout_{floor_name}.png'
        plt.savefig(output_path, bbox_inches='tight', pad_inches=0.3)
        plt.close()
        
        print(f"레이아웃 이미지 저장 완료: {output_path}")

def main():
    """메인 실행 함수"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="주차장 레이아웃 시각화 도구",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="layout_images",
        help="이미지 저장 디렉토리"
    )
    
    parser.add_argument(
        "--dpi",
        type=int,
        default=100,
        help="이미지 해상도 (DPI)"
    )
    
    parser.add_argument(
        "--cell-size",
        type=float,
        default=1.2,
        help="그리드 셀 크기"
    )
    
    parser.add_argument(
        "--font-size",
        type=float,
        default=10.0,
        help="텍스트 폰트 크기"
    )
    
    args = parser.parse_args()
    
    # 시각화 도구 초기화 및 실행
    visualizer = ParkingLayoutVisualizer(args.output_dir)
    visualizer.dpi = args.dpi
    visualizer.cell_size = args.cell_size
    visualizer.font_size = args.font_size
    
    # 모든 층 시각화
    visualizer.visualize_all_floors()

if __name__ == "__main__":
    main() 