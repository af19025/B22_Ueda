from scipy.stats import norm
import pandas as pd
import numpy as np
from sklearn.mixture import GaussianMixture
import os
from file_reader import reader 
import seaborn as sns
import math
from scipy import stats

# 閾値設定
congestion1 = 2.51  
congestion2 = 5.84  
congestion3 = 8.94
congestion4 = 11.24
theta = congestion4  # 必要に応じて変更
std = 2.22
mean = 0.83  # 非混雑時のRTTの平均

# 読み込むCSVファイル
#file_name = "rtt_data4.3.csv"
file_name = "rtt_data_evil4.1.csv"
directory_path = "/Users/tu/Documents/GitHub/B22_UEDA"
file_path = os.path.join(directory_path, file_name)

def clean_data(df, file_name):
    df['cleaned_column'] = pd.to_numeric(df.iloc[:, 1], errors='coerce')
    df = df.dropna(subset=['cleaned_column'])
    if df.empty:
        print(f"警告: {file_name} - クレンジング後のデータが空です。データの内容を確認してください。")
        return np.array([]).reshape(-1, 1), None
    return df['cleaned_column'].values.reshape(-1, 1), df['cleaned_column'].mean()

def cdf(extract_rtt):
    sns.set()
    
    # 基準となるデータ (通常時)
    normal_data = reader('rtt_data4.csv')
    normal_rtt = normal_data[:, 1]  # 2列目（RTT）のみ取得
    
    # データ型を数値に変換し、NaN を削除
    normal_rtt = pd.to_numeric(normal_rtt, errors='coerce')  # 数値変換
    normal_rtt = pd.Series(normal_rtt).dropna().values  # NaNを削除しNumPy配列に変換

    # **デバッグ用出力**
    print(f"normal_rtt type: {type(normal_rtt)}")
    print(f"normal_rtt sample: {normal_rtt[:10]}")  # 先頭10個を表示

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
    return 1 if extract_mean >= upper_bound else 0

data = reader(file_path)
df = pd.DataFrame(data)
X, mean_value = clean_data(df, file_name)

if X.size == 0:
    result = '無効なデータ'
else:
    gmm = GaussianMixture(n_components=2, random_state=0).fit(X)
    labels = gmm.predict(X)
    centers = gmm.means_
    gamma_upper = max(centers[:, 0])
    gamma_lower = min(centers[:, 0])
    y_axis = gamma_upper / gamma_lower
    print(f'{file_name} - 中心間の比：', y_axis)
    
    if y_axis <= theta:
        cdf_result = cdf(X.flatten())  # Xを1次元配列に変換
        result = '不正APを検知した' if cdf_result == 1 else '不正APは検知されなかった'
    else:
        result = '不正APを検知した'

print("\n判定結果:")
print(f"{file_name}: {result}")
