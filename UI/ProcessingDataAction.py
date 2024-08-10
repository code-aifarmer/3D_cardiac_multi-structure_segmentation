import sys

from PyQt5.QtWidgets import QApplication, QMainWindow
from PyQt5 import QtCore, QtWidgets, QtCore
# 导入login_window.py、my_main_window.py里面全部内容
import ProcessingData
import Read_Label
import TransLabelOrder
import UnzipData



class ReadLabel_window(Read_Label.Ui_MainWindow, QMainWindow):
    def __init__(self):
        super(ReadLabel_window, self).__init__()
        self.setupUi(self)
        self.pushButton.clicked.connect(self.close)


class Process_window(ProcessingData.Ui_MainWindow, QMainWindow):
    def __init__(self):
        super(Process_window, self).__init__()
        self.setupUi(self)
        self.pushButton.clicked.connect(self.close)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ProcessingData_Obj = ProcessingData()
    ProcessingData_Obj.show();
    ReadLabel_Obj = Read_Label()
    ProcessingData_Obj.pushButton.clicked.connect(ReadLabel_Obj.show)
    sys.exit(app.exec_())