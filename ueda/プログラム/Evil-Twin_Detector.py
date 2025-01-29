from scipy.stats import norm
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from file_reader import reader 
import os
from sklearn.mixture import GaussianMixture

# 閾値設定
not_congestion = 2.27  # 非混雑時の閾値
congestion = 5.84  # 混雑時の閾値
theta = congestion  # 必要に応じて変更
#std = 0.35
std = 2.22
mean = 0.83 #非混雑時のRTTの平均 1.6

# データを格納するリスト
results = []

# CSVファイルが格納されているディレクトリ
#directory_path = '/Users/tu/Documents/GitHub/B22_ueda_ver3/test_nor'
directory_path = '/Users/tu/Documents/GitHub/B22_ueda_ver3/test_atc'

# ディレクトリ内のCSVファイルを処理
for i in range(11, 21):  # test_nor1.csv ~ test_nor10.csv
    #file_name = f'test_nor{i}.csv'
    file_name = f'test_atc{i}.csv'
    file_path = os.path.join(directory_path, file_name)
    
    # データの読み込み
    data = reader(file_path)

    def clean_data(df):
        # 2列目のデータを数値に変換 (エラーが出た場合は NaN にする)
        df['cleaned_column'] = pd.to_numeric(df.iloc[:, 1], errors='coerce')
        # NaN を含む行を削除
        df = df.dropna(subset=['cleaned_column'])

        # もしデータがない場合、処理を続行せずに警告を表示する
        if df.empty:
            print(f"警告: {file_name} - クレンジング後のデータが空です。データの内容を確認してください。")
            return np.array([]).reshape(-1, 1)
        
        return df['cleaned_column'].values.reshape(-1, 1), df['cleaned_column'].mean()

    X, mean_value = clean_data(pd.DataFrame(data))

    if X.size == 0:
        print(f"{file_name} - 有効なデータがないため、処理を停止します。データのクレンジング条件を見直してください。")
        results.append((file_name, '無効なデータ'))
    else:
        # 平均値が1.3を超える場合のみ閾値判定を実行
        if mean_value > mean:
            # GMMクラスタリング
            gmm = GaussianMixture(n_components=2, random_state=0).fit(X)
            labels = gmm.predict(X)
            centers = gmm.means_

            # クラスタ中心を比較して、中心間の比を算出
            gamma_upper = max(centers[:, 0])
            gamma_lower = min(centers[:, 0])
            y_axis = gamma_upper / gamma_lower
            print(f'{file_name} - 中心間の比：', y_axis)
            
            # 閾値判定
            if y_axis <= theta :
                results.append((file_name, '不正APは検知されなかった'))
            else:
                results.append((file_name, '不正APを検知した'))
        else:
            results.append((file_name, '平均値が閾値以下のため、閾値判定をスキップしました'))

# 最終的な判定結果を表示
print("\n最終判定結果:")
for file_name, result in results:
    print(f"{file_name}: {result}")
