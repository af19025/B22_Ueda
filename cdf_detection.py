# 累積分布関数(CDF)を作成するプログラム
# 確率密度関数（推定）も出せる
# t分布を用いた区間推定を行っている
# k-meansを用いた検知で「不正APあり」と判断した際にこの検知を追加で行う

def cdf(extract_data):
        import numpy as np
        import matplotlib.pyplot as plt
        from scipy.stats import norm
        from math import pi, exp, sqrt
        import statistics
        from scipy.optimize import curve_fit
        from scipy.special import erf
        import seaborn as sns
        import math
        from scipy.stats import t
        #np.set_printoptions(threshold=np.inf) # 配列の表示最大数（inf = 無限）
        from sklearn.neighbors import KernelDensity
        from file_reader import reader # 通常時のデータ
        sns.set()
        from scipy import stats

        # normalファイル 
        # 検知する負荷に合わせてファイル名変更
        data1_1column = reader('test-Buffalo0.csv')

        # 2列目のみ吹き出す
        extract_data = extract_data[:, 1:2]
        #print(extract_data)
        # リストを配列(numpy)に変換 (※リストと配列は別物)
        #extract_data = np.array(extract_data)
        #print(extract_data)

        # n次元(今回は2次元)のNumpy配列を1次元のNumpy配列に変換する
        extract_data = extract_data.flatten()
        #print("extract_dataは",extract_data)

        #for i in range(len(extract_data)):
                #extract_data[i] = float(extract_data[i])

        # n行2列の配列から2列目のみ抜き出す
        data1 = data1_1column[:, 1:2]

        # 1次元配列に変換
        data1 = data1.flatten()
        #print("data1は",data1)

        #不偏分散
        var = np.var(data1)

        #平均
        mean = np.mean(data1)
        extract_mean = np.mean(extract_data)
        print('抜き出した100個のデータの平均：',extract_mean)
        #print('data1の平均：',mean)

        #自由度
        deg_of_freedom = len(data1) - 1

        #標準偏差（統計量tの分母）
        scale = math.sqrt(var/len(data1))

        # データのCDFを計算
        cdf = stats.cumfreq(data1, numbins=len(data1))

        # 信頼区間の計算
        confidence_interval = stats.t.interval(0.95, len(data1)-1, loc=np.mean(data1), scale=stats.sem(data1))

        # 信頼区間の下限と上限を取得
        lower_bound = confidence_interval[0]
        upper_bound = confidence_interval[1]

        print("95%信頼区間の下限:", lower_bound)
        print("95%信頼区間の上限:", upper_bound)

        if extract_mean <=lower_bound or extract_mean >= upper_bound :
                return 1

        else :
                return 0

        