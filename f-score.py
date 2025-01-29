def calculate_precision(tp, fp):
    if tp + fp == 0:
        return 0.0
    return tp / (tp + fp)

def calculate_recall(tp, fn):
    if tp + fn == 0:
        return 0.0
    return tp / (tp + fn)

def calculate_accuracy(tp, fp, fn, tn):
    total = tp + fp + fn + tn
    if total == 0:
        return 0.0
    return (tp + tn) / total

def f_score(precision, recall):
    if precision + recall == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)

# 混同行列の要素（例: TP=50, FP=10, FN=5, TN=100）
tp = 15  # True Positives: 正しく予測した正例の数
fp = 7  # False Positives: 誤って予測した正例の数
fn = 5   # False Negatives: 見逃した正例の数
tn = 13 # True Negatives: 正しく予測した負例の数

# 適合率・再現率・正解率・F値の計算
precision = calculate_precision(tp, fp)
recall = calculate_recall(tp, fn)
accuracy = calculate_accuracy(tp, fp, fn, tn)
f1 = f_score(precision, recall)

print(f'適合率 (Precision): {precision:.2f}')
print(f'再現率 (Recall): {recall:.2f}')
print(f'正解率 (Accuracy): {accuracy:.2f}')
print(f'F値 (F-score): {f1:.2f}')
