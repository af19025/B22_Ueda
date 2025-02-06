
import pandas as pd
import matplotlib.pyplot as plt

# CSVファイルを読み込む
file_path = "/Users/tu/Documents/GitHub/B22_Ueda/traffic_rtt_data2.csv"
df = pd.read_csv(file_path)

# 散布図を作成
plt.figure(figsize=(8, 6))
plt.scatter(df['Traffic (Mbps)'], df['RTT (ms)'], color='darkblue', alpha=1.0, s=20)

# 軸ラベルを設定
plt.xlabel("Average traffic in 10 seconds (Mbps)")
plt.ylabel("RTT in 10 seconds (ms)")

# 軸の目盛り間隔を設定
plt.xticks([96, 482, 944, 1304])
plt.yticks([i * 0.1 for i in range(11)])

# グリッドを追加
plt.grid(True, linestyle="--", alpha=0.6)

# グラフを表示
plt.show()

