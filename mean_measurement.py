import pandas as pd
import numpy as np

# クレンジングと平均計算を行う関数
def cleanse_and_calculate_mean(file_path):
    # CSVファイルを読み込み
    df = pd.read_csv(file_path)
    
    # 2列目のデータを取り出し
    column_data = df.iloc[:, 1]
    
    # 数値以外のデータを除去（クレンジング）
    column_data_clean = pd.to_numeric(column_data, errors='coerce').dropna()
    
    # 平均を計算
    return column_data_clean.mean()

# ファイルリスト
file_paths = [f'legal-que{i}.csv' for i in range(1, 11)]

# 各ファイルの平均を計算し、リストに格納
means = [cleanse_and_calculate_mean(file) for file in file_paths]

# 全ファイルの平均を計算
overall_mean = np.mean(means)

print("各ファイルの平均:", means)
print("全ファイルの平均:", overall_mean)
