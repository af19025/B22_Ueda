import numpy as np
import math
from scipy import stats
import seaborn as sns
from file_reader import reader  # 通常時のデータ

def cdf(extract_data_file):
    sns.set()
    
    # 基準となるデータ (通常時)
    normal_data = reader('rtt_data.csv')
    normal_rtt = normal_data[:, 1].flatten()  # 2列目（RTT）のみ取得
    
    # 入力データのRTT
    extract_data = reader(extract_data_file)
    extract_rtt = extract_data[:, 1].flatten()
    
    # 基準データの統計量
    mean = np.mean(normal_rtt)
    var = np.var(normal_rtt, ddof=1)  # 不偏分散
    std_err = math.sqrt(var / len(normal_rtt))  # 標準誤差
    
    # 95%信頼区間の計算 (t分布)
    confidence_interval = stats.t.interval(0.95, len(normal_rtt) - 1, loc=mean, scale=std_err)
    lower_bound, upper_bound = confidence_interval
    
    # 入力データの平均
    extract_mean = np.mean(extract_rtt)
    print(f'抜き出したデータの平均: {extract_mean}')
    print(f'95%信頼区間: ({lower_bound}, {upper_bound})')
    
    # 平均値が信頼区間外かを判定
    return 1 if extract_mean <= lower_bound or extract_mean >= upper_bound else 0
