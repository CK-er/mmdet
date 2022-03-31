# 通过朴素贝叶斯对鸢尾花数据进行分类

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB, GaussianNB
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

iris = datasets.load_iris() # 加载鸢尾花数据
iris_x = iris.data  # 获取数据
# print(iris_x)
iris_x = iris_x[:, :2]  # 取前两个特征值
# print(iris_x)
iris_y = iris.target    # 0， 1， 2
x_train, x_test, y_train, y_test = train_test_split(iris_x, iris_y, test_size=0.75, random_state=1) # 对数据进行分类 一部分最为训练一部分作为测试
# clf = GaussianNB()
# ir = clf.fit(x_train, y_train)
clf = Pipeline([
         ('sc', StandardScaler()),
         ('clf', GaussianNB())])     # 管道这个没深入理解 所以不知所以然
ir = clf.fit(x_train, y_train.ravel())  # 利用训练数据进行拟合

# 画图：
x1_max, x1_min = max(x_test[:, 0]), min(x_test[:, 0])   # 取0列特征得最大最小值
x2_max, x2_min = max(x_test[:, 1]), min(x_test[:, 1])   # 取1列特征得最大最小值
t1 = np.linspace(x1_min, x1_max, 500)   # 生成500个测试点
t2 = np.linspace(x2_min, x2_max, 500)
x1, x2 = np.meshgrid(t1, t2)  # 生成网格采样点
x_test1 = np.stack((x1.flat, x2.flat), axis=1)
y_hat = ir.predict(x_test1) # 预测
mpl.rcParams['font.sans-serif'] = [u'simHei']   # 识别中文保证不乱吗
mpl.rcParams['axes.unicode_minus'] = False
cm_light = mpl.colors.ListedColormap(['#77E0A0', '#FF8080', '#A0A0FF']) # 测试分类的颜色
cm_dark = mpl.colors.ListedColormap(['g', 'r', 'b'])    # 样本点的颜色
plt.figure(facecolor='w')
plt.pcolormesh(x1, x2, y_hat.reshape(x1.shape), cmap=cm_light)  # y_hat  25000个样本点的画图，
plt.scatter(x_test[:, 0], x_test[:, 1], edgecolors='k', s=50, c=y_test, cmap=cm_dark)   # 测试数据的真实的样本点（散点） 参数自行百度
plt.xlabel(u'花萼长度', fontsize=14)
plt.ylabel(u'花萼宽度', fontsize=14)
plt.title(u'GaussianNB对鸢尾花数据的分类结果', fontsize=18)
plt.grid(True)
plt.xlim(x1_min, x1_max)
plt.ylim(x2_min, x2_max)
plt.show()
y_hat1 = ir.predict(x_test)
result = y_hat1 == y_test
print(result)
acc = np.mean(result)
print('准确度: %.2f%%' % (100 * acc))