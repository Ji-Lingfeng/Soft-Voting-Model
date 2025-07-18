import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, roc_auc_score
import matplotlib.pyplot as plt
import numpy as np

# 读取数据
dataset = pd.read_csv("clean/总.csv")

# 提取特征
X = dataset.iloc[:, :-1]  # 假设最后一列是标签列

# 提取标签
y = dataset["是否血栓"]

scaler = StandardScaler()
X = scaler.fit_transform(X)  # 对所有数据进行归一化

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=45)

# 初始化模型
models = [
    ('lr', LogisticRegression(penalty='l2', C=1.0, solver='lbfgs', max_iter=1000)),  # 调整正则化类型、惩罚系数、求解器和最大迭代次数
    ('svc', SVC(kernel='rbf', C=1, probability=True, max_iter=-1)),  # 使用RBF核，调整惩罚系数，开启概率估计，不限制迭代次数
    ('rf', RandomForestClassifier(n_estimators=100, max_depth=None, random_state=45)),  # 增加决策树数量，不限制树深度，设定随机种子
    ('xgb', XGBClassifier(use_label_encoder=False, eval_metric='logloss', n_estimators=100, learning_rate=0.01)),
    # 调整树的数量、学习率
    ('nn', MLPClassifier(hidden_layer_sizes=(10,), activation='relu', solver='adam', alpha=0.0001, max_iter=1000))
    # 添加激活函数、正则化参数和优化器
]

# 计算每个模型的AUC
auc_scores = []
for name, model in models:
    scores = cross_val_score(model, X_train, y_train, cv=5, scoring='roc_auc')
    auc_scores.append(scores.mean())

# 归一化权重
weights = [auc / sum(auc_scores) for auc in auc_scores]

# 创建软投票集成模型，并分配权重
ensemble = VotingClassifier(estimators=models, voting='soft', weights=weights)

# 训练模型
ensemble.fit(X_train, y_train)

# 预测
y_pred_test = ensemble.predict(X_test)
y_pred_prob = ensemble.predict_proba(X_test)[:, 1]  # 获取预测概率

std_dev = np.std(y_pred_prob)
print(f"Predicted Probability Standard Deviation: {std_dev:.3f}")

# 计算混淆矩阵及评价指标
cm = confusion_matrix(y_true=y_test, y_pred=y_pred_test)
TP = cm[1, 1]
FP = cm[0, 1]
TN = cm[0, 0]
FN = cm[1, 0]

# 计算并输出准确率、敏感度和特异度
accuracy = (TP + TN) / (TP + FP + TN + FN)
sensitivity = TPR = TP / (TP + FN)
specificity = TNR = TN / (TN + FP)

print(f"True Positive (TP): {TP}")
print(f"False Positive (FP): {FP}")
print(f"True Negative (TN): {TN}")
print(f"False Negative (FN): {FN}")
print(f"Accuracy: {accuracy:.3f}")
print(f"Sensitivity (TPR): {sensitivity:.3f}")
print(f"Specificity (TNR): {specificity:.3f}")

# 计算原始的分类报告
report_dict = classification_report(y_true=y_test, y_pred=y_pred_test, labels=[0, 1], target_names=['未血栓', '血栓'],
                                    output_dict=True)

# 将字典转换为DataFrame以便格式化
report_df = pd.DataFrame(report_dict).transpose()


# 保留小数点后三位的格式化函数
def format_float(value):
    return f"{value:.3f}"


# 应用格式化函数到DataFrame中的所有浮点数列
for column in report_df.columns:
    if report_df[column].dtype == 'float64':
        report_df[column] = report_df[column].apply(format_float)

# 打印格式化后的报告
print(report_df.to_string(index=True))

# 计算AUC并绘制ROC曲线
fpr, tpr, _ = roc_curve(y_true=y_test, y_score=y_pred_prob)
auc_score = roc_auc_score(y_true=y_test, y_score=y_pred_prob)
print("AUC: %.3f" % auc_score)

# 绘制ROC曲线
plt.figure()
plt.plot(fpr, tpr, label=f'Soft Voting(AUC={auc_score:.3f})', color='orange')
plt.plot([0, 1], [0, 1], 'k--')  # 对角线
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('1-Specificity')
plt.ylabel('Sensitivity')
plt.legend(loc="lower right")
plt.show()
