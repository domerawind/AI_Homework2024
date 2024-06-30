from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, recall_score, f1_score
from dataPreparing import dataPrep

X_train, X_test, y_train, y_test = dataPrep('./TNSRE2014.xlsx')

# 创建SVM模型
svm_model = SVC(kernel='rbf', C=0.7, random_state=42 , cache_size=1000)

# 训练模型
svm_model.fit(X_train, y_train)

# 使用测试集进行预测
y_pred = svm_model.predict(X_test)

# 评估模型
accuracy = accuracy_score(y_test, y_pred)
recall = recall_score(y_test, y_pred, pos_label='acceptable')
f1 = f1_score(y_test, y_pred, pos_label='acceptable')

# 打印评估结果
print(f"SVM模型的准确率: {accuracy}")
print(f"SVM模型的召回率: {recall}")
print(f"SVM模型的F1分数: {f1}")
