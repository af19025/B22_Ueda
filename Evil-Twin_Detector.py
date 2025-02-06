from scipy.stats import norm
import pandas as pd
import numpy as np
from sklearn.mixture import GaussianMixture
import os
from file_reader import reader 

# 閾値設定
congestion1 = 2.51  
congestion2 = 5.84  
congestion3 = 8.94
congestion4 = 11.24
theta = congestion2  # 必要に応じて変更
std = 2.22
mean = 0.83  # 非混雑時のRTTの平均

# 読み込むCSVファイル
file_name = "rtt_data2.1.csv"
directory_path = "/Users/tu/Documents/GitHub/B22_UEDA"
file_path = os.path.join(directory_path, file_name)

def clean_data(df, file_name):
    df['cleaned_column'] = pd.to_numeric(df.iloc[:, 1], errors='coerce')
    df = df.dropna(subset=['cleaned_column'])
    if df.empty:
        print(f"警告: {file_name} - クレンジング後のデータが空です。データの内容を確認してください。")
        return np.array([]).reshape(-1, 1), None
    return df['cleaned_column'].values.reshape(-1, 1), df['cleaned_column'].mean()

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
        result = '不正APは検知されなかった'
    else:
        result = '不正APを検知した'

print("\n判定結果:")
print(f"{file_name}: {result}")
