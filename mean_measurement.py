import numpy as np
import pandas as pd
import os
from sklearn.cluster import KMeans
#3 12
#4 15
#
# データのクレンジング：数値に変換可能なデータのみを使用
def clean_data(df):
    df['cleaned_column'] = pd.to_numeric(df.iloc[:, 1], errors='coerce')
    df = df.dropna(subset=['cleaned_column'])
    return df['cleaned_column'].values.reshape(-1, 1)

# クラスタリングとy軸比を計算
def process_file(file_path):
    if not os.path.exists(file_path):
        print(f"ファイル {file_path} が見つかりません。")
        return

    data = pd.read_csv(file_path)
    X = clean_data(data)

    # k-meansクラスタリングの実行
    kmeans = KMeans(n_clusters=2, init='k-means++', random_state=0).fit(X)
    centers = kmeans.cluster_centers_

    # クラスタ中心の大きさを比較
    gamma_upper = max(centers[:, 0])
    gamma_lower = min(centers[:, 0])
    y_axis = gamma_upper / gamma_lower

    # 結果を有効数字3桁で表示
    print(f"{os.path.basename(file_path)} のクラスタ中心：", round(gamma_upper, 3), round(gamma_lower, 3))
    print(f"{os.path.basename(file_path)} のy軸比: {y_axis:.3f}")

# rtt_data_1.1.csv ファイルの処理を実行
file_path = 'rtt_data0.csv'
process_file(file_path)
