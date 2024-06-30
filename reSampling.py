import numpy as np
from sklearn.utils import resample
from dataPreparing import dataPrep

def resampleData():
    X_train, X_test, y_train, y_test = dataPrep('./TNSRE2014.xlsx')
    y_train = y_train.map({'acceptable': 1, 'unacceptable': 0})
    y_test = y_test.map({'acceptable': 1, 'unacceptable': 0})
    # 将训练数据拆分为正类和负类
    X_train_positive = X_train[y_train == 1]
    X_train_negative = X_train[y_train == 0]

    # 过采样负类数据
    X_train_negative_upsampled = resample(X_train_negative, 
                                        replace=True,     # 进行过采样
                                        n_samples=len(X_train_positive),    # 使正负类样本数相同
                                        random_state=42)

    # 重新组合过采样的数据集
    X_train_balanced = np.vstack((X_train_positive, X_train_negative_upsampled))
    y_train_balanced = np.hstack((np.ones(len(X_train_positive)), 
                                np.zeros(len(X_train_negative_upsampled))))

    return X_train_balanced, X_test, y_train_balanced, y_test
