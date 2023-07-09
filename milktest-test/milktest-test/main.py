import sys
from PyQt5 import QtCore, QtGui, QtWidgets
import qt

if __name__ == '__main__':
    # 获取UIC窗口操作权限
    app = QtWidgets.QApplication(sys.argv)

    MainWindow = QtWidgets.QMainWindow()
    # 调自定义的界面（即刚转换的.py对象）


    Ui = qt.Ui_MainWindow()
    # 这里也引用了一次.py文件的名字注意
    Ui.setupUi(MainWindow)
    # 显示窗口并释放资源

    MainWindow.show()
    sys.exit(app.exec_())

    #     self.pushButton.clicked.connect(self.cao1)
    #     self.pushButton_2.clicked.connect(self.cao2)
    #     # self.pushButton_3.clicked.connect(self.cao3)
    #     self.pushButton_4.clicked.connect(self.cao4)
    #
    #
    # def cao1(self):
    #     print("读取待测溶液的光强，并将获取的数据送入到预测模型中")
    #
    #
    #     x1 = self.lineEdit.text()
    #     x4 = self.lineEdit_2.text()
    #     x2 = self.lineEdit_3.text()
    #     x3 = self.lineEdit_4.text()
    #     x5 = self.lineEdit_5.text()
    #     x6 = self.lineEdit_6.text()
    #     print(u'x1: %s x2: %s x3: %s x4: %s x5: %s x6: %s' % (x1, x2, x3, x4, x5, x6))
    #     data = np.array([x1, x2, x3, x4, x5, x6])
    #     # 保存为a.csv
    #     with open('a.csv', 'w', newline='') as file:
    #         a = csv.writer(file, delimiter=',')
    #         a.writerow(data)
    #
    #     # 读取文件
    #     # with open('a.csv', 'r', newline='') as file:
    #     #     a = csv.reader(file, delimiter=',')
    #     #     for rows in a:
    #     #         print(rows)
    # def cao2(self):
    #     from sklearn import datasets, ensemble
    #     print("将送入到预测模型的数据进行预测，显示出牛奶蛋白质和脂肪的含量")
    #     pre_train = pd.read_csv('a.csv', encoding="unicode_escape")
    #     target1 = 'Protein'  # 定义目标值
    #     target2 = 'Fat'
    #     train = pre_train.dropna()
    #     X_columns = [x for x in train.columns]
    #     X_train = train[X_columns]
    #     y_train = train[target1]
    #
    #     regr = ensemble.GradientBoostingRegressor(n_estimators=90, max_depth=14, learning_rate=0.11, loss='ls')
    #     regr.fit(X_train, y_train)  # 训练模型
    #     pred_train = regr.predict(X_train)  # 输出预测值
    #     pred_test = regr.predict(y_train)  # 输出预测值
    #     print(pred_train)
    #     print(pred_test)
    #
    #
    #
    #
    # def cao3(self):
        # print("保存预测数据")
        # x1 = self.lineEdit.text()
        # x2 = self.lineEdit_3.text()
        # x3 = self.lineEdit_4.text()
        # x4 = self.lineEdit_2.text()
        # x5 = self.lineEdit_5.text()
        # x6 = self.lineEdit_6.text()
        # protain = self.lineEdit_8.text()
        # fat = self.lineEdit_7.text()
        # print(u'x1: %s x2: %s x3: %s x4: %s x5: %s x6: %s protain: %s fat: %s' % (x1, x2, x3, x4, x5, x6, protain, fat))
        # data2 = np.array([x1, x2, x3, x4, x5, x6, protain, fat])
        # # 保存为result.csv
        # with open('result.csv', 'w', newline='') as file:
        #     result = csv.writer(file, delimiter=',')
        #     result.writerow(data2)

    # def cao4(self):
    #     print("退出应用程序")
    #     app = QApplication.instance()
    #     app.quit()
