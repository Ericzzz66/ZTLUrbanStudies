from sklearn import svm
from sklearn.model_selection import train_test_split
import numpy as np
import os


# define converts(字典)
# def Iris_label(s):
#     it = {b'Iris-setosa': 0, b'Iris-versicolor': 1, b'Iris-virginica': 2}
#     return it[s]

# 1.读取数据集
path =  'your_path'
# converters={4:Iris_label}中“4”指的是第5列：将第5列的str转化为label(number)
data = np.loadtxt(path, dtype=float, delimiter=',')
# 特征的个数
features_number = 0

# 2.划分数据与标签
x, y = np.split(data, indices_or_sections=(features_number,), axis=1)  # x为数据，y为标签
# x = x[:, 0:8]#为便于后边画图显示，只选取前两维度。若不用画图，可选取前四列x[:,0:4]
train_data, test_data, train_label, test_label = train_test_split(x, y, random_state=30, train_size=0.85,test_size=0.15)  # sklearn.model_selection.
# print(train_data)
# 分数不为整数时转换
train_label = train_label.astype('int')
test_label = test_label.astype('int')

# 3.训练svm分类器
#   kernel='linear'时，为线性核，C越大分类效果越好，但有可能会过拟合（defaul C=1）。
#　 kernel='rbf'时（default），为高斯核，gamma值越小，分类界面越连续；gamma值越大，分类界面越“散”，分类效果越好，但有可能会过拟合。
#　 decision_function_shape='ovr'时，为one v rest（一对多），即一个类别与其他类别进行划分，
#　 decision_function_shape='ovo'时，为one v one（一对一），即将类别两两之间进行划分，用二分类的方法模拟多分类的结果。
classifier = svm.SVC(C=3000, kernel='rbf', gamma=2, decision_function_shape='ovo')
classifier.fit(train_data, train_label.ravel())  # ravel函数在降维时默认是行序优先


# 4.计算svc分类器的准确率
train_acc = classifier.score(train_data, train_label)
test_acc = classifier.score(test_data, test_label)
print("训练集：", train_acc)
print("测试集：", test_acc)


# 4.也可直接调用accuracy_score方法计算准确率
# from sklearn.metrics import accuracy_score
#
# tra_label = classifier.predict(train_data)  # 训练集的预测标签
# tes_label = classifier.predict(test_data)  # 测试集的预测标签
# train_acc = accuracy_score(train_label, tra_label)
# test_acc = accuracy_score(test_label, tes_label)
# print("训练集：", train_acc)
# print("测试集：", test_acc)

# 创建保存文件夹
folder = 'save_path'
if not os.path.exists(folder):
    os.mkdir('save_model')

# 5.保存训练结果
import joblib
joblib.dump(classifier, "save_model/{}.m".format(test_acc))
