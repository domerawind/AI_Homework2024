import numpy as np
from keras.src.models import Sequential
from keras.src.layers import LSTM, Dense
from keras.src.regularizers import L1, L2, L1L2
from sklearn.metrics import accuracy_score, recall_score, f1_score
from reSampling import resampleData

X_train, X_test, y_train, y_test = resampleData()
# 重塑输入数据为3D [样本数, 时间步长, 特征数]
X_train_lstm = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
X_test_lstm = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))

# 创建LSTM模型
lstm_model = Sequential()
lstm_model.add(LSTM(192, input_shape=(X_train_lstm.shape[1], X_train_lstm.shape[2]), activation='relu' ,kernel_regularizer=L1L2(0.001 , 0.005)))
lstm_model.add(Dense(32, activation='relu', kernel_regularizer=L2(0.001)))
lstm_model.add(Dense(1, activation='sigmoid'))

# 编译模型
lstm_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# 训练模型
lstm_model.fit(X_train_lstm, y_train, epochs=20, batch_size=25, verbose=1)

# 使用测试集进行预测
y_pred_prob = lstm_model.predict(X_test_lstm)
y_pred = (y_pred_prob > 0.75).astype(int).flatten()

# 评估模型
accuracy = accuracy_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# 打印评估结果
print(f"LSTM模型的准确率: {accuracy}")
print(f"LSTM模型的召回率: {recall}")
print(f"LSTM模型的F1分数: {f1}")
