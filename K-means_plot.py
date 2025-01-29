import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import os

# データのクレンジング：数値に変換可能なデータのみを使用
def clean_data(df):
    df['cleaned_column'] = pd.to_numeric(df.iloc[:, 1], errors='coerce')
    df = df.dropna(subset=['cleaned_column'])
    return df['cleaned_column'].values.reshape(-1, 1)

# クラスタリングとプロット
def process_file(file_path, label_position, color, marker_color):
    if not os.path.exists(file_path):
        print(f"ファイル {file_path} が見つかりません。")
        return

    data = pd.read_csv(file_path)
    X = clean_data(data)

    # k-meansクラスタリングの実行
    kmeans = KMeans(n_clusters=2, init='k-means++', random_state=0).fit(X)
    centers = kmeans.cluster_centers_
    labels = kmeans.labels_

    # クラスタ中心の大きさを比較
    gamma_upper = max(centers[:, 0])
    gamma_lower = min(centers[:, 0])
    y_axis = gamma_upper / gamma_lower
    print(f"{os.path.basename(file_path)} のクラスタ中心：", gamma_upper, gamma_lower)
    print(f"{os.path.basename(file_path)} のy軸比:", y_axis)

    # データと中心点をプロット
    plt.scatter(np.full(len(X[:, 0]), label_position), X[:, 0], c=labels, cmap=color)
    plt.scatter(np.full(len(centers[:, 0]), label_position), centers[:, 0], marker='x', s=200, linewidths=3, color=marker_color)

# グラフ全体の設定とファイルの処理実行
def plot_clusters():
    # グラフの設定
    fig = plt.figure()
    plt.ylabel("RTT (ms)")
    plt.xlim(0, 0.3)
    plt.xticks([0, 0.1, 0.2, 0.3], ["", "legitimate", "rogue", ""])

    # 特定のファイルを指定して処理
    files_to_process = [
        ('test_nor/test_nor1.csv', 0.1, 'viridis', 'r'),  # test_atcフォルダ内
        ('test_nor/test_nor11.csv', 0.2, 'plasma', 'b')  # training dataフォルダ内
    ]

    for file_path, label_position, color, marker_color in files_to_process:
        process_file(file_path, label_position, color, marker_color)

    plt.show()

# グラフのプロットを実行
plot_clusters()
