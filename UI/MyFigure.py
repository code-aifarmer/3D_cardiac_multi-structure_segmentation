import matplotlib

matplotlib.use("Qt5Agg")  # 声明使用QT5
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


class MyFigure(FigureCanvas):
    def __init__(self, width=5, height=4, dpi=100):
        # 创建一个创建Figure
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        # 在父类中激活Figure窗口
        super(MyFigure, self).__init__(self.fig)  # 此句必不可少，否则不能显示图形
        # 创建一个子图，用于绘制图形用，111表示子图编号，如matlab的subplot(1,1,1)
        self.axes = self.fig.add_subplot(111)