##################################
# cdf作成に必要なデータ抽出プログラム #
##################################

def reader(filename):
    from array import array
    from re import L
    from scipy.stats import norm
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn import cluster

    import seaborn as sns
    sns.set()


    # csvファイルを2次元配列として格納
    study = pd.read_csv(filename,header=None)
    
    # 欠損値が一つでも含まれる行が削除
    study = study.dropna(how='any')

    study = study.values
        
    return (study)