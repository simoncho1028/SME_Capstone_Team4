import os
import subprocess
import pandas as pd
import matplotlib.pyplot as plt
import glob
import matplotlib.font_manager as fm
import platform
from typing import Dict

# 한글 폰트 설정
if platform.system() == 'Darwin':  # macOS
    plt.rcParams['font.family'] = 'AppleGothic'
else:  # Linux or Windows
    plt.rcParams['font.family'] = 'NanumGothic'

plt.rcParams['axes.unicode_minus'] = False  # 마이너스 기호 깨짐 방지

def run_simulation(charger_count, sim_time=86400):
    # 시뮬레이션 실행
    cmd = f"python main.py --charger {charger_count} --time {sim_time}"
    subprocess.run(cmd, shell=True)
    
    # 가장 최근 결과 디렉토리 찾기
    results_dirs = glob.glob(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results_sim_*'))
    if not results_dirs:
        raise Exception("시뮬레이션 결과를 찾을 수 없습니다.")
    latest_dir = max(results_dirs, key=os.path.getctime)
    
    # 가장 최근 요약 파일 찾기
    summary_files = glob.glob(os.path.join(latest_dir, 'simulation_summary_*.txt'))
    if not summary_files:
        raise Exception("시뮬레이션 요약 파일을 찾을 수 없습니다.")
    latest_summary = max(summary_files, key=os.path.getctime)
    
    # 요약 파일에서 지표 추출
    metrics = extract_metrics(latest_summary)
    metrics['charger_count'] = charger_count
    
    return metrics

def extract_metrics(summary_file: str) -> Dict[str, float]:
    metrics = {}
    with open(summary_file, 'r', encoding='utf-8') as f:
        content = f.read()
        
        # 충전 실패율
        if '충전 실패율:' in content:
            failure_rate = float(content.split('충전 실패율:')[1].split('%')[0].strip())
            metrics['charging_failure_rate'] = failure_rate
        
        # 일반차 이동거리
        if '평균 distance (일반 차량):' in content:
            normal_distance = float(content.split('평균 distance (일반 차량):')[1].split('칸')[0].strip())
            metrics['avg_distance_normal'] = normal_distance
        
        # EV 이동거리
        if '평균 distance (EV):' in content:
            ev_distance = float(content.split('평균 distance (EV):')[1].split('칸')[0].strip())
            metrics['avg_distance_ev'] = ev_distance
        
        # 일반차 만차율
        if '일반차 만차 시간:' in content:
            full_time = float(content.split('일반차 만차 시간:')[1].split('분')[0].strip())
            metrics['full_parking_rate'] = full_time
        
        # 설치비용 (충전소 개수 * 1000)
        if '충전소 수:' in content:
            charger_count = int(content.split('충전소 수:')[1].split('개')[0].strip())
            metrics['installation_cost'] = charger_count * 1000
    
    return metrics

def normalize_metrics(metrics_df: pd.DataFrame) -> pd.DataFrame:
    # 각 지표별 min-max 정규화
    normalized_df = metrics_df.copy()
    for col in ['charging_failure_rate', 'avg_distance_normal', 'avg_distance_ev', 'full_parking_rate', 'installation_cost']:
        min_val = metrics_df[col].min()
        max_val = metrics_df[col].max()
        normalized_df[f'norm_{col}'] = (metrics_df[col] - min_val) / (max_val - min_val)
    
    # 목적함수 계산 (정규화된 지표들의 합)
    normalized_df['objective_function'] = normalized_df[[
        'norm_charging_failure_rate',
        'norm_avg_distance_normal',
        'norm_avg_distance_ev',
        'norm_full_parking_rate',
        'norm_installation_cost'
    ]].sum(axis=1)
    
    return normalized_df

def plot_results(normalized_df: pd.DataFrame):
    plt.figure(figsize=(12, 6))
    
    # 각 지표별 라인 플롯
    plt.plot(normalized_df['charger_count'], normalized_df['norm_charging_failure_rate'], 
             label='충전 실패율', marker='o')
    plt.plot(normalized_df['charger_count'], normalized_df['norm_avg_distance_normal'], 
             label='일반차 이동거리', marker='s')
    plt.plot(normalized_df['charger_count'], normalized_df['norm_avg_distance_ev'], 
             label='EV 이동거리', marker='^')
    plt.plot(normalized_df['charger_count'], normalized_df['norm_full_parking_rate'], 
             label='일반차 만차율', marker='d')
    plt.plot(normalized_df['charger_count'], normalized_df['norm_installation_cost'], 
             label='설치비용', marker='*')
    plt.plot(normalized_df['charger_count'], normalized_df['objective_function'], 
             label='목적함수', marker='x', linewidth=2)
    
    plt.xlabel('충전소 수')
    plt.ylabel('정규화된 값')
    plt.title('충전소 수에 따른 정규화된 지표 변화')
    plt.legend()
    plt.grid(True)
    
    # 결과 저장
    plt.savefig('simulation_results.png')
    plt.close()

def main():
    # 충전소 개수 리스트 (5개와 14~35개)
    charger_counts = [5] + list(range(14, 36))
    
    # 각 충전소 개수에 대해 시뮬레이션 실행
    all_metrics = []
    for charger_count in charger_counts:
        print(f"\n충전소 {charger_count}개 시뮬레이션 실행 중...")
        metrics = run_simulation(charger_count, sim_time=2592000)  # 30일
        all_metrics.append(metrics)
    
    # 결과를 DataFrame으로 변환
    metrics_df = pd.DataFrame(all_metrics)
    
    # 14~35개 결과만 포함하는 DataFrame 생성
    metrics_df_14_35 = metrics_df[metrics_df['charger_count'] >= 14].copy()
    
    # 정규화 및 시각화 (14~35개 결과만 사용)
    normalized_df = normalize_metrics(metrics_df_14_35)
    
    # 결과 저장
    normalized_df.to_csv('simulation_metrics.csv', index=False)
    plot_results(normalized_df)
    
    print("시뮬레이션 결과가 simulation_metrics.csv와 simulation_results.png에 저장되었습니다.")

if __name__ == "__main__":
    main() 