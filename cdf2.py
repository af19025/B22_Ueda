import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def plot_cdf(csv_file1, csv_file2):
    # CSVファイル1を読み込む
    data1 = pd.read_csv(csv_file1, header=None, names=['Index', 'RTT'])
    data1['RTT'] = pd.to_numeric(data1['RTT'], errors='coerce')  # 数値型に変換
    rtt_data1 = data1['RTT'].dropna()  # NaNを除去
    sorted_rtt1 = np.sort(rtt_data1)
    cdf1 = np.arange(1, len(sorted_rtt1) + 1) / len(sorted_rtt1)

    # CSVファイル2を読み込む
    data2 = pd.read_csv(csv_file2, header=None, names=['Index', 'RTT'])
    data2['RTT'] = pd.to_numeric(data2['RTT'], errors='coerce')  # 数値型に変換
    rtt_data2 = data2['RTT'].dropna()  # NaNを除去
    sorted_rtt2 = np.sort(rtt_data2)
    cdf2 = np.arange(1, len(sorted_rtt2) + 1) / len(sorted_rtt2)

    # 90%信頼区間を計算
    lower_bound1 = np.percentile(sorted_rtt1, 5)  # 5%点
    upper_bound1 = np.percentile(sorted_rtt1, 95)  # 95%点

    lower_bound2 = np.percentile(sorted_rtt2, 5)  # 5%点
    upper_bound2 = np.percentile(sorted_rtt2, 95)  # 95%点

    # 信頼区間の幅を計算
    ci_width1 = upper_bound1 - lower_bound1
    ci_width2 = upper_bound2 - lower_bound2

    # CDFをプロット
    plt.figure(figsize=(8, 6))
    plt.plot(sorted_rtt1, cdf1, marker='.', linestyle='none', label='CDF 1')
    plt.plot(sorted_rtt2, cdf2, marker='.', linestyle='none', label='CDF 2')

    # 信頼区間をプロット
    plt.axvline(x=lower_bound1, color='blue', linestyle='--', label='90% CI (CDF 1) Lower Bound')
    plt.axvline(x=upper_bound1, color='blue', linestyle='--', label='90% CI (CDF 1) Upper Bound')

    plt.axvline(x=lower_bound2, color='orange', linestyle='--', label='90% CI (CDF 2) Lower Bound')
    plt.axvline(x=upper_bound2, color='orange', linestyle='--', label='90% CI (CDF 2) Upper Bound')

    plt.title('CDF of RTT Data with 90% Confidence Interval')
    plt.xlabel('RTT (ms)')
    plt.ylabel('CDF')
    plt.grid(True)

    # x軸の目盛りを調整
    plt.xticks(np.linspace(min(sorted_rtt1.min(), sorted_rtt2.min()), max(sorted_rtt1.max(), sorted_rtt2.max()), num=10))

    # 信頼区間の値と幅を表示
    print(f"90%信頼区間 (CDF 1): 下限 = {lower_bound1:.2f}, 上限 = {upper_bound1:.2f}, 幅 = {ci_width1:.2f}")
    print(f"90%信頼区間 (CDF 2): 下限 = {lower_bound2:.2f}, 上限 = {upper_bound2:.2f}, 幅 = {ci_width2:.2f}")

    plt.legend()
    plt.show()

# 使用例
# 2つのCSVファイルのパスを指定して関数を呼び出してください。
plot_cdf('rtt_data3.csv', 'rtt_data_evil3.csv')
