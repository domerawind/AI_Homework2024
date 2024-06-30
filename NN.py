import tensorflow as tf
from keras import Sequential
from keras.src.layers import Dense
from keras.src.regularizers import L2
from sklearn.metrics import accuracy_score, recall_score, f1_score
from reSampling import resampleData

X_train, X_test, y_train, y_test = resampleData()


# 创建神经网络模型
nn_model = Sequential()
nn_model.add(Dense(256, input_dim=X_train.shape[1], activation='relu', kernel_regularizer=L2(0.01)))
nn_model.add(Dense(128, activation='relu'))
nn_model.add(Dense(64, activation='relu'))
nn_model.add(Dense(1, activation='sigmoid'))

# 编译模型
nn_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# 训练模型
nn_model.fit(X_train, y_train, epochs=40, batch_size=30, verbose=1)

# 使用测试集进行预测
y_pred_prob = nn_model.predict(X_test)
y_pred = (y_pred_prob > 0.5).astype(int).flatten()

# 评估模型
accuracy = accuracy_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# 打印评估结果
print(f"神经网络模型的准确率: {accuracy}")
print(f"神经网络模型的召回率: {recall}")
print(f"神经网络模型的F1分数: {f1}")
