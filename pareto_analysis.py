import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import platform
import matplotlib.font_manager as fm
from mpl_toolkits.mplot3d import Axes3D

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

# 파레토 최적해 출력
print("\n파레토 최적해 (이동거리 제외):")
print(f"충전소 수: {pareto_df['charger_count'].tolist()}")
print("\n지표:")
print(pareto_df[['charger_count'] + metrics].to_string(index=False))

# 2D 시각화 (설치비용 vs 충전 실패율)
plt.figure(figsize=(10, 6))
plt.scatter(df['installation_cost'], df['charging_failure_rate'], 
           alpha=0.5, label='비 최적해')
plt.scatter(pareto_df['installation_cost'], pareto_df['charging_failure_rate'],
           color='red', label='파레토 최적해')
plt.xlabel('설치비용 (원)')
plt.ylabel('충전 실패율 (%)')
plt.title('파레토 프론트 (설치비용 vs 충전 실패율)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('pareto_front_2d.png', dpi=300, bbox_inches='tight')
plt.close()

# 3D 시각화 (충전 실패율 vs 만차율 vs 설치비용)
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# 비 최적해
ax.scatter(df['charging_failure_rate'], 
          df['full_parking_rate'],
          df['installation_cost'],
          alpha=0.3, label='비 최적해')

# 파레토 최적해
ax.scatter(pareto_df['charging_failure_rate'],
          pareto_df['full_parking_rate'],
          pareto_df['installation_cost'],
          color='red', label='파레토 최적해')

ax.set_xlabel('충전 실패율 (%)')
ax.set_ylabel('만차율 (분)')
ax.set_zlabel('설치비용 (원)')
ax.set_title('파레토 프론트 (3D)')
plt.legend()
plt.savefig('pareto_front_3d.png', dpi=300, bbox_inches='tight')
plt.close()

print("\n시각화 결과가 다음 파일에 저장되었습니다:")
print("1. pareto_front_2d.png - 2D 파레토 프론트")
print("2. pareto_front_3d.png - 3D 파레토 프론트") 