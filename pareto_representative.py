import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import platform
import matplotlib.font_manager as fm

# 한글 폰트 설정
if platform.system() == 'Darwin':  # macOS
    plt.rc('font', family='AppleGothic')
else:  # Windows/Linux
    plt.rc('font', family='NanumGothic')
plt.rcParams['axes.unicode_minus'] = False

# 데이터 로드
df = pd.read_csv('simulation_metrics.csv')

# 파레토 최적해 계산
def is_pareto_efficient(costs):
    is_efficient = np.ones(costs.shape[0], dtype=bool)
    for i, c in enumerate(costs):
        if is_efficient[i]:
            is_efficient[is_efficient] = np.any(costs[is_efficient] < c, axis=1)
            is_efficient[i] = True
    return is_efficient

# 원본 지표 선택 (이동거리 제외)
metrics = ['charging_failure_rate', 'full_parking_rate', 'installation_cost']
costs = df[metrics].values
pareto_mask = is_pareto_efficient(costs)
pareto_df = df[pareto_mask].copy()

# 5개의 대표적인 해 선택 (충전소 수 기준으로 균등하게 분포)
charger_counts = pareto_df['charger_count'].values
min_chargers = min(charger_counts)
max_chargers = max(charger_counts)
step = (max_chargers - min_chargers) / 4

selected_indices = []
for target in np.arange(min_chargers, max_chargers + step, step):
    idx = np.abs(charger_counts - target).argmin()
    selected_indices.append(idx)

representative_df = pareto_df.iloc[selected_indices].copy()
representative_df = representative_df.sort_values('charger_count')

# 결과를 CSV 파일로 저장
representative_df.to_csv('pareto_representative_solutions.csv', index=False)

# 시각화
plt.figure(figsize=(15, 15))

# 1. 충전 실패율 그래프
plt.subplot(3, 1, 1)
x = np.arange(len(representative_df))
width = 0.8
plt.bar(x, representative_df['charging_failure_rate'], width,
        color='#1f77b4')
plt.xlabel('충전소 수')
plt.ylabel('충전 실패율 (%)')
plt.title('충전 실패율 비교')
plt.xticks(x, [f'{int(count)}개' for count in representative_df['charger_count']])
plt.grid(True, alpha=0.3)

# 2. 만차 시간 그래프
plt.subplot(3, 1, 2)
plt.bar(x, representative_df['full_parking_rate'], width,
        color='#ff7f0e')
plt.xlabel('충전소 수')
plt.ylabel('만차 시간 (분)')
plt.title('만차 시간 비교')
plt.xticks(x, [f'{int(count)}개' for count in representative_df['charger_count']])
plt.grid(True, alpha=0.3)

# 3. 설치비용 그래프
plt.subplot(3, 1, 3)
plt.bar(x, representative_df['installation_cost'], width,
        color='#2ca02c')
plt.xlabel('충전소 수')
plt.ylabel('설치비용 (천원)')
plt.title('설치비용 비교')
plt.xticks(x, [f'{int(count)}개' for count in representative_df['charger_count']])
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('pareto_representative_comparison.png', dpi=300, bbox_inches='tight')
plt.close()

# 추가: 레이더 차트 (정규화된 값으로)
plt.figure(figsize=(10, 10))
ax = plt.subplot(111, polar=True)

# 각 지표의 최대값으로 정규화
max_values = representative_df[metrics].max()
normalized_values = representative_df[metrics] / max_values

# 레이더 차트를 위한 각도 계산
angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False)
angles = np.concatenate((angles, [angles[0]]))  # 닫힌 다각형을 위해

colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
for i, (idx, row) in enumerate(normalized_values.iterrows()):
    values = row.values
    values = np.concatenate((values, [values[0]]))  # 닫힌 다각형을 위해
    plt.plot(angles, values, 'o-', linewidth=2, 
             label=f'충전소 {int(representative_df.iloc[i]["charger_count"])}개',
             color=colors[i])
    plt.fill(angles, values, alpha=0.1, color=colors[i])

plt.xticks(angles[:-1], ['충전 실패율', '만차 시간', '설치비용'])
plt.ylim(0, 1)
plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
plt.title('정규화된 지표 비교 (레이더 차트)')

plt.tight_layout()
plt.savefig('pareto_representative_radar.png', dpi=300, bbox_inches='tight')
plt.close()

print("분석이 완료되었습니다.")
print("결과가 다음 파일에 저장되었습니다:")
print("1. pareto_representative_solutions.csv - 선택된 5개 해의 상세 지표")
print("2. pareto_representative_comparison.png - 각 지표별 막대 그래프")
print("3. pareto_representative_radar.png - 정규화된 지표 비교 레이더 차트") 