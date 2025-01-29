import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
import statistics
from sklearn.preprocessing import StandardScaler

# 処理するCSVファイル名を指定
file_path = "rtt_data3.csv"  # ここに指定するCSVファイル名を記述

# y軸比と全体データ用のリストを初期化
y_axis_list = []
all_values = []  # すべてのファイルの2列目データを格納するリスト

try:
    # CSVファイルを読み込み
    data = pd.read_csv(file_path, header=None)

    # 数値データのみをフィルタリング
    data = data.apply(pd.to_numeric, errors='coerce')

    # 特定の列（ここでは2列目）に注目し、欠損値がない行のみを抽出
    data = data.dropna(subset=[1])  # 2列目（index 1）に欠損値がある行を削除

    if data.empty:
        print(f'ファイル {file_path} に有効なデータがありません')
    else:
        # 2列目のデータをall_valuesリストに追加
        all_values.extend(data.iloc[:, 1].values)

        # DBSCANクラスタリングを実行
        X = data.iloc[:, 1].values.reshape(-1, 1)

        # DBSCANを適用
        dbscan = DBSCAN(eps=0.5, min_samples=5)
        labels = dbscan.fit_predict(X)

        # DBSCANの結果に基づき、クラスタが2つ以上の場合に処理
        if len(set(labels)) > 1:
            # クラスタのデータを抽出
            cluster_1 = X[labels == 0]
            cluster_2 = X[labels == 1]

            # クラスタの中心を計算
            gamma_upper = max(np.mean(cluster_1), np.mean(cluster_2))
            gamma_lower = min(np.mean(cluster_1), np.mean(cluster_2))

            # クラスタの中心間の比を計算
            y_axis = gamma_upper / gamma_lower
            y_axis_list.append(y_axis)  # y軸比をリストに追加
            print(f'{file_path} の中心間の比は {round(y_axis, 2)} です')
        else:
            print(f'{file_path} に有効な2つのクラスタがありません')

    # 全体統計を計算
    if all_values:
        mean_all_values = np.mean(all_values)
        std_dev_all_values = np.std(all_values)

        print("\n全ファイルの2列目データの平均:", round(mean_all_values, 2))
        print("全ファイルの2列目データの標準偏差:", round(std_dev_all_values, 2))

    # y軸比の統計情報を計算
    if y_axis_list:
        mean_y_axis = sum(y_axis_list) / len(y_axis_list)
        variance_y_axis = statistics.variance(y_axis_list) if len(y_axis_list) > 1 else 0
        std_dev_y_axis = statistics.stdev(y_axis_list) if len(y_axis_list) > 1 else 0

        print("全てのy軸比の平均:", round(mean_y_axis, 2))
        print("全てのy軸比の分散:", round(variance_y_axis, 2))
        print("全てのy軸比の標本標準偏差:", round(std_dev_y_axis, 2))
    else:
        print("有効なy軸比が計算できませんでした。")

except FileNotFoundError:
    print(f"指定されたファイル {file_path} が見つかりません。")
except Exception as e:
    print("エラーが発生しました:", str(e))
