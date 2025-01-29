import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def plot_cdf(csv_file):
    # CSVファイルを読み込む
    data = pd.read_csv(csv_file, header=None, names=['Index', 'RTT'])
    data['RTT'] = pd.to_numeric(data['RTT'], errors='coerce')  # 数値型に変換
    rtt_data = data['RTT'].dropna()  # NaNを除去
    sorted_rtt = np.sort(rtt_data)
    cdf = np.arange(1, len(sorted_rtt) + 1) / len(sorted_rtt)

    # 0.9信頼区間を計算（5%と95%パーセンタイル）
    lower_bound = np.percentile(sorted_rtt, 5)  # 5%点
    upper_bound = np.percentile(sorted_rtt, 95)  # 95%点

    # 信頼区間の表示
    print(f"90%信頼区間: {lower_bound} ms - {upper_bound} ms")
    
    # 最大と最小の差を表示
    range_diff = upper_bound - lower_bound
    print(f"90%信頼区間の最大と最小の差: {range_diff} ms")

    # CDFをプロット
    plt.figure(figsize=(8, 6))
    plt.plot(sorted_rtt, cdf, marker='.', linestyle='none')
    plt.xlabel('RTT (ms)', fontsize=14)  # x軸ラベルのフォントサイズを大きく
    plt.ylabel('CDF', fontsize=14)  # y軸ラベルのフォントサイズを大きく
    plt.grid(True)

    # x軸の目盛りを0.5刻みで設定
    # minとmaxの間に0.5刻みで目盛りを表示
    min_rtt = np.floor(min(sorted_rtt) / 0.5) * 0.5
    max_rtt = np.ceil(max(sorted_rtt) / 0.5) * 0.5
    plt.xticks(np.arange(min_rtt, max_rtt + 0.5, step=0.5), fontsize=12)  # メモリのフォントサイズを大きく
    plt.yticks(fontsize=12)  # y軸のメモリのフォントサイズを大きく

    # 信頼区間をプロット
    plt.axvline(x=lower_bound, color='green', linestyle='--', label='90% CI Lower Bound')
    plt.axvline(x=upper_bound, color='red', linestyle='--', label='90% CI Upper Bound')

    plt.legend(fontsize=12)  # 凡例のフォントサイズを大きく
    
    # 余白を調整
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)  # 余白を少なくする

    plt.show()

# 使用例
# 'rtt_data1.csv' のみを指定して関数を呼び出してください。
plot_cdf('rtt_data1.csv')
