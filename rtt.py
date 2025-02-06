#0.147, 0.26
#0.197, 0.263 1
#0.354, 0.525 2
#0.624, 0.708 3
#0.68, 0.721 4
import numpy as np
import pandas as pd

# パラメータ
mean_rtt_target = 0.2  # 平均RTT
std_dev_rtt_target = 0.25  # 標準偏差
alpha = 2.0  # パレート分布の形状パラメータ 元は：1.5
x_m_base = 0.01  # 最小値
num_samples = 300  # サンプル数
tolerance = 0.001  # 許容誤差

# 初期化
rtt_samples = (np.random.pareto(alpha, num_samples) + 1) * x_m_base

# 平均と標準偏差を反復調整
for i in range(100):  # 最大100回の反復
    # 現在の平均と標準偏差
    current_mean = np.mean(rtt_samples)
    current_std_dev = np.std(rtt_samples)

    # 平均と標準偏差が目標値に収束したら終了
    if abs(current_mean - mean_rtt_target) < tolerance and abs(current_std_dev - std_dev_rtt_target) < tolerance:
        break

    # 平均を調整
    scale_factor_mean = mean_rtt_target / current_mean
    rtt_samples *= scale_factor_mean

    # 標準偏差を調整
    current_std_dev = np.std(rtt_samples)  # 再計算
    scale_factor_std = std_dev_rtt_target / current_std_dev
    rtt_samples *= scale_factor_std

# 最終調整後の平均と標準偏差
final_mean = np.mean(rtt_samples)
final_std_dev = np.std(rtt_samples)

# 小数点以下3桁に丸める
rtt_samples = np.round(rtt_samples, 3)

# データフレーム作成とCSV保存
df = pd.DataFrame({"number of packet": range(1, num_samples + 1), "RTT(ms)": rtt_samples})
df.to_csv("rtt_data5.csv", index=False)

print(f"生成された300個のRTTデータの平均: {final_mean:.3f}")
print(f"生成された300個のRTTデータの標準偏差: {final_std_dev:.3f}")
print("CSVファイルが作成されました。")
