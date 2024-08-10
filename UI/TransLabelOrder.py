# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'TransLabelOrder.ui'
#
# Created by: PyQt5 UI code generator 5.15.9
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(800, 514)
        MainWindow.setStyleSheet("/*\n"
"ElegantDark Style Sheet for QT Applications\n"
"Author: Jaime A. Quiroga P.\n"
"Company: GTRONICK\n"
"Last updated: 17/04/2018\n"
"Available at: https://github.com/GTRONICK/QSS/blob/master/ElegantDark.qss\n"
"*/\n"
"QMainWindow {\n"
"    background-color:rgb(82, 82, 82);\n"
"}\n"
"QTextEdit {\n"
"    background-color:rgb(42, 42, 42);\n"
"    color: rgb(0, 255, 0);\n"
"}\n"
"QPushButton{\n"
"    border-style: outset;\n"
"    border-width: 2px;\n"
"    border-top-color: qlineargradient(spread:pad, x1:0.5, y1:0.6, x2:0.5, y2:0.4, stop:0 rgba(115, 115, 115, 255), stop:1 rgba(62, 62, 62, 255));\n"
"    border-right-color: qlineargradient(spread:pad, x1:0.4, y1:0.5, x2:0.6, y2:0.5, stop:0 rgba(115, 115, 115, 255), stop:1 rgba(62, 62, 62, 255));\n"
"    border-left-color: qlineargradient(spread:pad, x1:0.6, y1:0.5, x2:0.4, y2:0.5, stop:0 rgba(115, 115, 115, 255), stop:1 rgba(62, 62, 62, 255));\n"
"    border-bottom-color: rgb(58, 58, 58);\n"
"    border-bottom-width: 1px;\n"
"    border-style: solid;\n"
"    color: rgb(255, 255, 255);\n"
"    padding: 2px;\n"
"    background-color: qlineargradient(spread:pad, x1:0.5, y1:1, x2:0.5, y2:0, stop:0 rgba(77, 77, 77, 255), stop:1 rgba(97, 97, 97, 255));\n"
"}\n"
"QPushButton:hover{\n"
"    border-style: outset;\n"
"    border-width: 2px;\n"
"    border-top-color: qlineargradient(spread:pad, x1:0.5, y1:0.6, x2:0.5, y2:0.4, stop:0 rgba(180, 180, 180, 255), stop:1 rgba(110, 110, 110, 255));\n"
"    border-right-color: qlineargradient(spread:pad, x1:0.4, y1:0.5, x2:0.6, y2:0.5, stop:0 rgba(180, 180, 180, 255), stop:1 rgba(110, 110, 110, 255));\n"
"    border-left-color: qlineargradient(spread:pad, x1:0.6, y1:0.5, x2:0.4, y2:0.5, stop:0 rgba(180, 180, 180, 255), stop:1 rgba(110, 110, 110, 255));\n"
"    border-bottom-color: rgb(115, 115, 115);\n"
"    border-bottom-width: 1px;\n"
"    border-style: solid;\n"
"    color: rgb(255, 255, 255);\n"
"    padding: 2px;\n"
"    background-color: qlineargradient(spread:pad, x1:0.5, y1:1, x2:0.5, y2:0, stop:0 rgba(107, 107, 107, 255), stop:1 rgba(157, 157, 157, 255));\n"
"}\n"
"QPushButton:pressed{\n"
"    border-style: outset;\n"
"    border-width: 2px;\n"
"    border-top-color: qlineargradient(spread:pad, x1:0.5, y1:0.6, x2:0.5, y2:0.4, stop:0 rgba(62, 62, 62, 255), stop:1 rgba(22, 22, 22, 255));\n"
"    border-right-color: qlineargradient(spread:pad, x1:0.4, y1:0.5, x2:0.6, y2:0.5, stop:0 rgba(115, 115, 115, 255), stop:1 rgba(62, 62, 62, 255));\n"
"    border-left-color: qlineargradient(spread:pad, x1:0.6, y1:0.5, x2:0.4, y2:0.5, stop:0 rgba(115, 115, 115, 255), stop:1 rgba(62, 62, 62, 255));\n"
"    border-bottom-color: rgb(58, 58, 58);\n"
"    border-bottom-width: 1px;\n"
"    border-style: solid;\n"
"    color: rgb(255, 255, 255);\n"
"    padding: 2px;\n"
"    background-color: qlineargradient(spread:pad, x1:0.5, y1:1, x2:0.5, y2:0, stop:0 rgba(77, 77, 77, 255), stop:1 rgba(97, 97, 97, 255));\n"
"}\n"
"QPushButton:disabled{\n"
"    border-style: outset;\n"
"    border-width: 2px;\n"
"    border-top-color: qlineargradient(spread:pad, x1:0.5, y1:0.6, x2:0.5, y2:0.4, stop:0 rgba(115, 115, 115, 255), stop:1 rgba(62, 62, 62, 255));\n"
"    border-right-color: qlineargradient(spread:pad, x1:0.4, y1:0.5, x2:0.6, y2:0.5, stop:0 rgba(115, 115, 115, 255), stop:1 rgba(62, 62, 62, 255));\n"
"    border-left-color: qlineargradient(spread:pad, x1:0.6, y1:0.5, x2:0.4, y2:0.5, stop:0 rgba(115, 115, 115, 255), stop:1 rgba(62, 62, 62, 255));\n"
"    border-bottom-color: rgb(58, 58, 58);\n"
"    border-bottom-width: 1px;\n"
"    border-style: solid;\n"
"    color: rgb(0, 0, 0);\n"
"    padding: 2px;\n"
"    background-color: qlineargradient(spread:pad, x1:0.5, y1:1, x2:0.5, y2:0, stop:0 rgba(57, 57, 57, 255), stop:1 rgba(77, 77, 77, 255));\n"
"}\n"
"QLineEdit {\n"
"    border-width: 1px; border-radius: 4px;\n"
"    border-color: rgb(58, 58, 58);\n"
"    border-style: inset;\n"
"    padding: 0 8px;\n"
"    color: rgb(255, 255, 255);\n"
"    background:rgb(100, 100, 100);\n"
"    selection-background-color: rgb(187, 187, 187);\n"
"    selection-color: rgb(60, 63, 65);\n"
"}\n"
"QLabel {\n"
"    color:rgb(255,255,255);    \n"
"}\n"
"QProgressBar {\n"
"    text-align: center;\n"
"    color: rgb(240, 240, 240);\n"
"    border-width: 1px; \n"
"    border-radius: 10px;\n"
"    border-color: rgb(58, 58, 58);\n"
"    border-style: inset;\n"
"    background-color:rgb(77,77,77);\n"
"}\n"
"QProgressBar::chunk {\n"
"    background-color: qlineargradient(spread:pad, x1:0.5, y1:0.7, x2:0.5, y2:0.3, stop:0 rgba(87, 97, 106, 255), stop:1 rgba(93, 103, 113, 255));\n"
"    border-radius: 5px;\n"
"}\n"
"QMenuBar {\n"
"    background:rgb(82, 82, 82);\n"
"}\n"
"QMenuBar::item {\n"
"    color:rgb(223,219,210);\n"
"    spacing: 3px;\n"
"    padding: 1px 4px;\n"
"    background: transparent;\n"
"}\n"
"\n"
"QMenuBar::item:selected {\n"
"    background:rgb(115, 115, 115);\n"
"}\n"
"QMenu::item:selected {\n"
"    color:rgb(255,255,255);\n"
"    border-width:2px;\n"
"    border-style:solid;\n"
"    padding-left:18px;\n"
"    padding-right:8px;\n"
"    padding-top:2px;\n"
"    padding-bottom:3px;\n"
"    background:qlineargradient(spread:pad, x1:0.5, y1:0.7, x2:0.5, y2:0.3, stop:0 rgba(87, 97, 106, 255), stop:1 rgba(93, 103, 113, 255));\n"
"    border-top-color: qlineargradient(spread:pad, x1:0.5, y1:0.6, x2:0.5, y2:0.4, stop:0 rgba(115, 115, 115, 255), stop:1 rgba(62, 62, 62, 255));\n"
"    border-right-color: qlineargradient(spread:pad, x1:0.4, y1:0.5, x2:0.6, y2:0.5, stop:0 rgba(115, 115, 115, 255), stop:1 rgba(62, 62, 62, 255));\n"
"    border-left-color: qlineargradient(spread:pad, x1:0.6, y1:0.5, x2:0.4, y2:0.5, stop:0 rgba(115, 115, 115, 255), stop:1 rgba(62, 62, 62, 255));\n"
"    border-bottom-color: rgb(58, 58, 58);\n"
"    border-bottom-width: 1px;\n"
"}\n"
"QMenu::item {\n"
"    color:rgb(223,219,210);\n"
"    background-color:rgb(78,78,78);\n"
"    padding-left:20px;\n"
"    padding-top:4px;\n"
"    padding-bottom:4px;\n"
"    padding-right:10px;\n"
"}\n"
"QMenu{\n"
"    background-color:rgb(78,78,78);\n"
"}\n"
"QTabWidget {\n"
"    color:rgb(0,0,0);\n"
"    background-color:rgb(247,246,246);\n"
"}\n"
"QTabWidget::pane {\n"
"        border-color: rgb(77,77,77);\n"
"        background-color:rgb(101,101,101);\n"
"        border-style: solid;\n"
"        border-width: 1px;\n"
"        border-radius: 6px;\n"
"}\n"
"QTabBar::tab {\n"
"    padding:2px;\n"
"    color:rgb(250,250,250);\n"
"      background-color: qlineargradient(spread:pad, x1:0.5, y1:1, x2:0.5, y2:0, stop:0 rgba(77, 77, 77, 255), stop:1 rgba(97, 97, 97, 255));\n"
"    border-style: solid;\n"
"    border-width: 2px;\n"
"      border-top-right-radius:4px;\n"
"   border-top-left-radius:4px;\n"
"    border-top-color: qlineargradient(spread:pad, x1:0.5, y1:0.6, x2:0.5, y2:0.4, stop:0 rgba(115, 115, 115, 255), stop:1 rgba(95, 92, 93, 255));\n"
"    border-right-color: qlineargradient(spread:pad, x1:0.4, y1:0.5, x2:0.6, y2:0.5, stop:0 rgba(115, 115, 115, 255), stop:1 rgba(95, 92, 93, 255));\n"
"    border-left-color: qlineargradient(spread:pad, x1:0.6, y1:0.5, x2:0.4, y2:0.5, stop:0 rgba(115, 115, 115, 255), stop:1 rgba(95, 92, 93, 255));\n"
"    border-bottom-color: rgb(101,101,101);\n"
"}\n"
"QTabBar::tab:selected, QTabBar::tab:last:selected, QTabBar::tab:hover {\n"
"      background-color:rgb(101,101,101);\n"
"      margin-left: 0px;\n"
"      margin-right: 1px;\n"
"}\n"
"QTabBar::tab:!selected {\n"
"        margin-top: 1px;\n"
"        margin-right: 1px;\n"
"}\n"
"QCheckBox {\n"
"    color:rgb(223,219,210);\n"
"    padding: 2px;\n"
"}\n"
"QCheckBox:hover {\n"
"    border-radius:4px;\n"
"    border-style:solid;\n"
"    padding-left: 1px;\n"
"    padding-right: 1px;\n"
"    padding-bottom: 1px;\n"
"    padding-top: 1px;\n"
"    border-width:1px;\n"
"    border-color: rgb(87, 97, 106);\n"
"    background-color:qlineargradient(spread:pad, x1:0.5, y1:0.7, x2:0.5, y2:0.3, stop:0 rgba(87, 97, 106, 150), stop:1 rgba(93, 103, 113, 150));\n"
"}\n"
"QCheckBox::indicator:checked {\n"
"    border-radius:4px;\n"
"    border-style:solid;\n"
"    border-width:1px;\n"
"    border-color: rgb(180,180,180);\n"
"      background-color:qlineargradient(spread:pad, x1:0.5, y1:0.7, x2:0.5, y2:0.3, stop:0 rgba(87, 97, 106, 255), stop:1 rgba(93, 103, 113, 255));\n"
"}\n"
"QCheckBox::indicator:unchecked {\n"
"    border-radius:4px;\n"
"    border-style:solid;\n"
"    border-width:1px;\n"
"    border-color: rgb(87, 97, 106);\n"
"      background-color:rgb(255,255,255);\n"
"}\n"
"QStatusBar {\n"
"    color:rgb(240,240,240);\n"
"}")
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.centralwidget)
        self.verticalLayout.setObjectName("verticalLayout")
        self.label = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(22)
        self.label.setFont(font)
        self.label.setStyleSheet("color: rgb(255, 255, 255);")
        self.label.setAlignment(QtCore.Qt.AlignCenter)
        self.label.setObjectName("label")
        self.verticalLayout.addWidget(self.label)
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.pushButton = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton.setStyleSheet("")
        self.pushButton.setObjectName("pushButton")
        self.horizontalLayout.addWidget(self.pushButton)
        self.lineEdit = QtWidgets.QLineEdit(self.centralwidget)
        self.lineEdit.setObjectName("lineEdit")
        self.horizontalLayout.addWidget(self.lineEdit)
        self.verticalLayout.addLayout(self.horizontalLayout)
        self.Trans_textEdit = QtWidgets.QTextEdit(self.centralwidget)
        self.Trans_textEdit.setObjectName("Trans_textEdit")
        self.verticalLayout.addWidget(self.Trans_textEdit)
        self.pushButton_2 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_2.setStyleSheet("")
        self.pushButton_2.setObjectName("pushButton_2")
        self.verticalLayout.addWidget(self.pushButton_2)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 800, 20))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.label.setText(_translate("MainWindow", "转换标签顺序"))
        self.pushButton.setText(_translate("MainWindow", "选择标签所在目录路径"))
        self.pushButton_2.setText(_translate("MainWindow", "开始转换"))
