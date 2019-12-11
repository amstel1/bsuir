
import sys
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QSlider
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import (QMainWindow, QAction, qApp, QApplication, QPushButton, QDesktopWidget,QVBoxLayout,QSlider,
                            QLabel,QRadioButton, QFileDialog, QWidget, QGridLayout,QButtonGroup, QHBoxLayout, QMenu, QSizePolicy, QMessageBox, QWidget, QLineEdit, QSpinBox)
from PyQt5.QtGui import QIcon
from PyQt5.QtCore import QSize, QRect 
from win32api import GetSystemMetrics
from PyQt5.QtCore import QSize, QRect 
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtWidgets import QMainWindow, QLabel, QGridLayout, QWidget, QComboBox
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import random
from sklearn.base import BaseEstimator
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from scipy.io.wavfile import read

import matplotlib.pyplot as plt
import numpy as np
import wave
import sys
sns.set_style('darkgrid')
plt.style.use('ggplot')
CURRENT_VERSION = 0.1

class Example(QMainWindow):

    def __init__(self):
        super().__init__()
        self.equation = 'No model fiited yet. Intending to fit y ~ a + b*x'
        self.initUI()
        
    def initUI(self):
        self.setWindowTitle('test')
        window_width = GetSystemMetrics(0)
        window_height = GetSystemMetrics(1)
        self.resize(0.7 * window_width, 0.7 * window_height)
        self.center()
        self.setWindowIcon(QIcon('Icon.png'))

        #inits
        self.openDirectoryDialog = ""
        self.data = np.empty(shape=(1,2), dtype=np.float)
        self.setMinimumSize(QSize(640, 480))    
        self.setWindowTitle("Регрессионный анализ одномерных данных") 
        centralWidget = QWidget(self)          
        self.setCentralWidget(centralWidget)   
       
        #Exit on menubar
        exitAct = QAction('&Exit', self)
        exitAct.setShortcut('Ctrl+Q')
        exitAct.setStatusTip('Exit applicatiion')
        exitAct.triggered.connect(qApp.quit)

        #Open on menubar
        openAct = QAction('&Open', self)
        openAct.setShortcut('Ctrl+O')
        openAct.setStatusTip('Open Directory')
        openAct.triggered.connect(self.openFile)
        openAct.triggered.connect(lambda: plotCan.plot_loaded(self.data))

        #menubar
        menubar = self.menuBar()
        fileMenu = menubar.addMenu('&File')
        fileMenu.addAction(exitAct)
        fileMenu.addAction(openAct)

        #Central
        #centralwidget = QWidget(self)
        #self.setCentralWidget(centralwidget)

        #Grid
        #grid = QGridLayout(centralwidget)
        #self.setLayout(grid)

        #Plot
        plotCan = PlotCanvas(self, width=10.88, height=7.48)
        #grid.addWidget(plotCan, 0,1)
        plotCan.move(250,0)

        #Закон распределения СВ
        self.label_4 = QtWidgets.QLabel(self)
        self.label_4.setGeometry(QtCore.QRect(10, 475, 110, 25))
        self.label_4.setText("Выбор закона СВ")
        #Закон распределения СВ
        self.comboBox = QtWidgets.QComboBox(self)
        self.comboBox.setGeometry(QtCore.QRect(10, 500, 110, 25))
        self.comboBox.addItems(['Нормальный','Равномерный',])
        self.comboBox.activated[str].connect(self.rename_sv_params)
        #self.comboBox.activated[str].connect(self.show_params)
        

        #Способ добавления СВ
        self.label = QtWidgets.QLabel(self)
        self.label.setGeometry(QtCore.QRect(130, 475, 110, 25))
        self.label.setText("Способ доб. СВ")
        #Способ добавления СВ
        self.comboBox2 = QtWidgets.QComboBox(self)
        self.comboBox2.setGeometry(QtCore.QRect(130, 500, 110, 25))
        self.comboBox2.addItems(['None','Additive','Multiplicative'])
        #self.comboBox2.activated[str].connect(self.show_params)
        #parameter1 СВ
        self.doubleSpinBox = QtWidgets.QDoubleSpinBox(self)
        self.doubleSpinBox.setGeometry(QtCore.QRect(45, 530, 75, 25))
        self.doubleSpinBox.setSingleStep(0.01)
        #parameter1 СВ
        self.label_2 = QtWidgets.QLabel(self)
        self.label_2.setGeometry(QtCore.QRect(10, 530, 34, 24))
        self.label_2.setText("μ")

        #parameter2 СВ
        self.doubleSpinBox_2 = QtWidgets.QDoubleSpinBox(self)
        self.doubleSpinBox_2.setGeometry(QtCore.QRect(45, 560, 75, 25))
        self.doubleSpinBox_2.setValue(1)
        self.doubleSpinBox_2.setSingleStep(0.01)
        #parameter2 СВ
        self.label_3 = QtWidgets.QLabel(self)
        self.label_3.setGeometry(QtCore.QRect(10, 560, 34, 24))
        self.label_3.setText("σ")
        
        self.slider  = QSlider(Qt.Horizontal,self)
        self.slider.setGeometry(QtCore.QRect(150, 530, 70, 30))
        self.slider.setMinimum(0)
        self.slider.setMaximum(100)
        self.slider.setValue(90)
        self.slider.setTickPosition(QSlider.TicksBelow)
        self.slider.setTickInterval(20)

        self.label_5 = QtWidgets.QLabel(self)
        self.label_5.setGeometry(QtCore.QRect(130, 560, 100, 24))
        self.label_5.setText("Случайность: 10%")
        self.label_6 = QtWidgets.QLabel(self)
        self.label_6.setGeometry(QtCore.QRect(130, 590, 100, 24))
        self.label_6.setText("Опред-ть: 90%")
        self.slider.valueChanged.connect(self.valuechange)

    
        #button
        #btn = QPushButton("Load Data", self)
        #btn.move(100,200)
        #btn.setVisible(True)
        #btn.resize(btn.sizeHint())
        #grid.addWidget(btn, 1,0)
        #btn.clicked.connect(lambda: plotCan.plot(self.data)) #good
        
        self.btn5 = QPushButton("Построить тренд",self)
        self.btn5.move(120, 350)
        self.btn5.resize(125,25)
        self.btn5.clicked.connect(lambda: plotCan.plot_determined(self.slider, self.comboBox, 
         self.comboBox2, self.doubleSpinBox, self.doubleSpinBox_2,self.combo1, self.spin1, self.param1, self.param2, self.param3))


        self.spin1 = QSpinBox(self)
        self.spin1.move(8,350)
        self.spin1.resize(100,25)
        self.spin1.setRange(5,9999)
        self.spin1.setValue(100)
        
        
        self.param1_label = QLabel(self)
        self.param1_label.move(8, 375)
        self.param1_label.setText("a")

        self.param1 = QLineEdit(self)
        self.param1.move(8, 400)    
        self.param1.resize(50,20)
        self.param1.setText('1')

        self.param2_label = QLabel(self)
        self.param2_label.move(65, 375)
        self.param2_label.setText("b")
        self.param2 = QLineEdit(self)
        self.param2.move(65, 400)    
        self.param2.resize(50,20)
        self.param2.setText('1')

        self.param3_label = QLabel(self)
        self.param3_label.move(138, 375)
        self.param3_label.setText("c")
        self.param3_label.resize(0, 0)
        self.param3 = QLineEdit(self)
        self.param3.move(138, 400)    
        self.param3.resize(0,0)
        self.param3.setText('1')


        self.combo1 = QComboBox(self)
        self.combo1.move(8,300)
        self.combo1.resize(150,25)
        self.combo1.addItems(['Линейный','Параболический','Гиперболический','Экспоненциальный','Показательный','Степенной','Логарифмический','Гармонический'])
        self.combo1.activated[str].connect(self.show_params)


        #label
        self.eq_label = QLabel(self)
        self.eq_label.move(8,30)
        #self.eq_label.setText('Fit')
        self.eq_label.setGeometry(QtCore.QRect(25, 0, 500, 100)) #(x, y, width, height)
        self.eq_label.setText("y=a+b*t")

        #combobox
        self.combo = QComboBox(self)
        self.combo.move(8,70)
        self.combo.resize(200,25)
        self.combo.addItems(['Линейный','Параболический','Гиперболический','Экспоненциальный','Показательный','Степенной','Логарифмический','Гармонический'])
        self.combo.activated[str].connect(self.combobox_changed)

        self.btn2 = QPushButton('Fit Regression',self)
        self.btn2.move(8, 160)
        self.btn2.resize(200,30)
        self.btn2.setText('Оценить линейный тренд')
        self.btn2.setEnabled(False)
        self.btn2.clicked.connect(self.fit_regression)
        #self.btn2.clicked.connect(lambda: plotCan.plot(self.data))
        self.btn2.clicked.connect(lambda: plotCan.plot_fitted(self.estimator, self.combo, self.data))     #plot fitted

        btn3 = QPushButton('Open file', self)
        btn3.move(8, 100)
        btn3.resize(200,30)
        btn3.clicked.connect(lambda: plotCan.clear(self))
        btn3.clicked.connect(self.openFile)
        btn3.clicked.connect(lambda: plotCan.clear(self))

        btn3.clicked.connect(lambda: plotCan.plot_loaded(self.data))
        
        #btn3.clicked.connect(lambda: plotCan.plot(self.data))

        btn4 = QPushButton('Clear', self)
        btn4.move(8, 130)
        btn4.resize(200,30)
        btn4.clicked.connect(lambda: plotCan.clear(self))

        self.show()

    def valuechange(self):
        s_val = int(self.slider.value())
        self.label_5.setText(f'Случайность: {100 - s_val}%')
        self.label_6.setText(f'Опред-ть: {s_val}%') 

    def onClick(self):
        self.radiobutton = self.sender()

    def center(self):
        qr = self.frameGeometry()
        cp = QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())

    def openFile(self):
        try:
            self.filepath = QFileDialog.getOpenFileName(self, "Open File",
            filter="text files (*.txt);;Excel files (*.xlsx);;""WAV files (*.wav);;Binary files (*.bin)")[0]
            if self.filepath.endswith('txt'):
                self.data = pd.read_csv(self.filepath, delimiter='/n', header=None, names=['y'])
                self.data['t'] = [x for x in range(1,len(self.data)+1)]
                print('csv')
                print(type(self.data))
            elif self.filepath.endswith('xlsx'):
                self.data = pd.read_excel(self.filepath, header=None, names = ['y'])
                self.data['t'] = [x for x in range(1,len(self.data)+1)]
                print('excel')
                print(type(self.data))
            elif self.filepath.endswith('wav'):
                spf = wave.open(self.filepath, "r")
                # Extract Raw Audio from Wav File
                signal = spf.readframes(-1)
                signal = np.fromstring(signal, "Int16")
                fs = spf.getframerate()
                # If Stereo
                if spf.getnchannels() == 2:
                    print("Just mono files")
                    sys.exit(0)
                Time = np.linspace(0, len(signal) / fs, num=len(signal))
                df = pd.DataFrame(np.concatenate((Time.reshape(signal.shape[0],1), (signal.reshape(signal.shape[0],1))), axis = 1))
                #rename df columns
                self.data = df[df[0] != 0]
                self.data = self.data.rename(columns = {0:'t', 1:'y'})
                print('wav')
                print(self.data.shape)
            try:
                if len(self.data) > 1:
                    self.btn2.setEnabled(True)
            except:
                self.btn2.setEnabled(False)
        except OSError:
            pass

    def buttonpress(self):
        self.plot(self.data)

    def combobox_changed(self, text):
        if text == 'Линейный':
            self.eq_label.setText("y=a+b*t")
            self.btn2.setText('Оценить линейный тренд')
        elif text == 'Логарифмический':
            self.eq_label.setText("y=a+b*ln(t)")
            self.btn2.setText('Оценить логарифмический тренд')
        elif text == 'Параболический':
            self.eq_label.setText("y=a+b*t+c*t^2")
            self.btn2.setText('Оценить параболический тренд')
        elif text == 'Гиперболический':
            self.eq_label.setText("y=a+b/t")
            self.btn2.setText('Оценить гиперболический тренд')
        elif text == 'Экспоненциальный':
            self.eq_label.setText("y=a*e^bt")
            self.btn2.setText('Оценить экспоненциальный тренд')
        elif text == 'Показательный':
            self.eq_label.setText("y=a*b^t")
            self.btn2.setText('Оценить показательный тренд')
        elif text == 'Степенной':
            self.eq_label.setText("y=a*t^b")
            self.btn2.setText('Оценить стпенной тренд')
        elif text == 'Гармонический':
            self.eq_label.setText("y=a*cos(pi*b*t+c)")
            self.btn2.setText('Оценить гармонический тренд')
        print(text)
    
    def show_params(self, text):
        if text == 'Линейный':
            self.param1.resize(50,20)
            self.param2.resize(50,20)
            self.param3.resize(0,0)
            self.param3_label.resize(0, 0)
        elif text == 'Параболический':
            self.param1.resize(50,20)
            self.param2.resize(50,20)
            self.param3.resize(50,20)
            self.param3_label.resize(25, 25)
        elif text == 'Гиперболический':
            self.param1.resize(50,20)
            self.param2.resize(50,20)
            self.param3.resize(0,0)
            self.param3_label.resize(0, 0)
        elif text == 'Экспоненциальный':
            self.param1.resize(50,20)
            self.param2.resize(50,20)
            self.param3.resize(0,0)
            self.param3_label.resize(0, 0)
        elif text == 'Показательный':
            self.param1.resize(50,20)
            self.param2.resize(50,20)
            self.param3.resize(0,0)
            self.param3_label.resize(0, 0)
        elif text == 'Степенной':
            self.param1.resize(50,20)
            self.param2.resize(50,20)
            self.param3.resize(0,0)
            self.param3_label.resize(0, 0)
        elif text == 'Логарифмический':
            self.param1.resize(50,20)
            self.param2.resize(50,20)
            self.param3.resize(0,0)
            self.param3_label.resize(0, 0)
        elif text == 'Гармонический':
            self.param1.resize(50,20)
            self.param2.resize(50,20)
            self.param3.resize(50,20)
            self.param3_label.resize(25, 25)

    def rename_sv_params(self,text):
        if text =='Нормальный':
            self.label_2.setText("μ")
            self.label_3.setText("σ")
        elif text =='Равномерный':
            self.label_2.setText("low")
            self.label_3.setText("high")

    def fit_regression(self):
        try:
            self.estimator = LinearRegression()
            X = self.data.loc[:, "t"].values.reshape(-1,1)
            y = self.data.loc[:, "y"].values.reshape(-1,1)
            if self.combo.currentText() =='Линейный':
                fitted_estimator = self.estimator.fit(X = X, y = y)
                y_pred = fitted_estimator.intercept_[0] + fitted_estimator.coef_[0][0] * X
                self.equation = f'y = {round(fitted_estimator.intercept_[0],2)} + {round(fitted_estimator.coef_[0][0],2)} * x;'
            elif self.combo.currentText() =='Логарифмический':
                fitted_estimator = self.estimator.fit(X = np.log(X), y = y)
                y_pred = fitted_estimator.intercept_[0] + fitted_estimator.coef_[0][0] * np.log(X)
                self.equation = f'y = {round(fitted_estimator.intercept_[0],2)} + {round(fitted_estimator.coef_[0][0],2)} * ln(x);'
            elif self.combo.currentText() == 'Параболический':
                X2 =self.data.loc[:, "t"].values.reshape(-1,1)**2
                fitted_estimator = self.estimator.fit(X = np.hstack((X, X2)), y = y)
                y_pred = fitted_estimator.intercept_[0] + fitted_estimator.coef_[0][0] * X + fitted_estimator.coef_[-1][-1] * X2
                self.equation = f'y = {round(fitted_estimator.intercept_[0],2)} + {round(fitted_estimator.coef_[0][0],2)} * x + {round(fitted_estimator.coef_[-1][-1],2)} * x^2;' 
            elif self.combo.currentText() == 'Гиперболический':
                fitted_estimator = self.estimator.fit(X = 1/X, y = y)
                y_pred = fitted_estimator.intercept_[0] + fitted_estimator.coef_[0][0] * 1/X
                self.equation = f'y = {round(fitted_estimator.intercept_[0],2)} + {round(fitted_estimator.coef_[0][0],2)}/x;'
            elif self.combo.currentText() == 'Экспоненциальный':
                #y=a*e^(b*t)
                #ln(y) = ln(a) + b*t
                #Y=A+b*t
                y = np.log(self.data.loc[:, "y"].values.reshape(-1,1))
                fitted_estimator = self.estimator.fit(X = X, y = y)
                A = fitted_estimator.intercept_[0]
                b = fitted_estimator.coef_[0][0]
                y_pred = np.exp(A) * np.exp(b*X)

                a = round(np.exp(A),2)
                b = round(fitted_estimator.coef_[0][0],2)                
                y = np.exp(y)
                self.equation = f'y = {a} * e^{b}x;'
            elif self.combo.currentText() == 'Показательный':
                #y=a*b^t
                #ln(y) = ln(a) + ln(b) * t
                #Y=A+B*t
                y = np.log(self.data.loc[:, "y"].values.reshape(-1,1))
                fitted_estimator = self.estimator.fit(X = X, y = y)
                A = fitted_estimator.intercept_[0]
                B = fitted_estimator.coef_[0][0]
                a = np.exp(A)
                b = np.exp(B)
                y_pred = A + B*X
                y_pred = np.exp(y_pred)
                y = np.exp(y)
                self.equation = f'y = {round(a,2)} * {round(b,2)}^x;'
            elif self.combo.currentText() == 'Степенной':
                #y=a*t^b
                #ln(y) = ln(a) + b*ln(t)
                #Y=A+b*T
                y = np.log(self.data.loc[:, "y"].values.reshape(-1,1))
                X = np.log(self.data.loc[:, "t"].values.reshape(-1,1))
                fitted_estimator = self.estimator.fit(X = X, y = y)
                A = fitted_estimator.intercept_[0]
                b = fitted_estimator.coef_[0][0]
                a = np.exp(A)
                y_pred = a*(np.exp(X)**b)
                #y_pred = np.exp(y_pred)
                y = np.exp(y)
                self.equation = f'y = {round(a,2)} * x^{round(b,2)};' 
                pass
            elif self.combo.currentText() == 'Гармонический':
                pass
            self.equation +=  '\n'
            self.equation +=  f'RMSE = {round(mean_squared_error(y_true = y, y_pred = y_pred)**0.5,2)}' 
            self.eq_label.setText(self.equation)
        except AttributeError:
            pass

class PlotCanvas(FigureCanvas):

    def __init__(self, parent=None, width=3, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        FigureCanvas.__init__(self, fig)
        self.setParent(parent)
        FigureCanvas.updateGeometry(self)
        FigureCanvas.setSizePolicy(self,
                QSizePolicy.Expanding,
                QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)
        sns.set_style('darkgrid')
        plt.style.use('ggplot')
        #self.plot()

    def clear(self, data = np.empty(shape=(1,2))):
        #self.matplotlibwidget.axes.clear()
        #self.matplotlibwidget.axes.figure.clear()
        #self.matplotlibwidget.axes.figure.canvas.draw_idle()
        try:
            self.ax.clear()
        except AttributeError:
            pass
        self.draw()
    
    def plot(self, data = np.empty(shape=(1,2))):
        sns.set_style('darkgrid')
        plt.style.use('ggplot')
        ax = self.figure.add_subplot(111)
        try:
            ax.plot(data.loc[:,0],data.loc[:,1], 'r-')
        except:
            ax.plot(data[:,0],data[:,1], 'r-')
        ax.set_title('PyQt Matplotlib Example')
        self.draw()

    def plot_determined(self,slider, zakon_sv, sposob_sv, mu_sv, sigma_sv,  trend_type, n_, a_, b_, c_):
        sns.set_style('darkgrid')
        plt.style.use('ggplot')
        self.ax = self.figure.add_subplot(1,1,1)
        opred = int(slider.value())/100
        sluch = (100 - int(slider.value()))/100
        n = int(n_.value())
        t = np.array([x for x in range(1, n+1)])
        #generate
        if zakon_sv.currentText() == 'Нормальный':
            e = np.random.normal(mu_sv.value(), sigma_sv.value(), n)
        elif zakon_sv.currentText() == 'Равномерный':
            e = np.random.uniform(mu_sv.value(), sigma_sv.value(), n)
        if trend_type.currentText() == 'Линейный':
            if a_.text().replace("-","").isnumeric() and b_.text().replace("-","").isnumeric():
                a = float(a_.text())
                b = float(b_.text())
                if sposob_sv.currentText() == 'None':
                    y = a + b*t
                elif sposob_sv.currentText() == 'Additive':
                    y = (a + b*t)*opred + e*sluch
                elif sposob_sv.currentText() == 'Multiplicative':
                    y = (a + b*t)*e
                self.ax.plot(t, y, marker='>', color='red', linewidth=2, alpha=0.4, label="Линейный")  # 
                self.ax.set_title("Title", fontsize=12, fontweight=0)
                self.ax.set_xlabel("xlabel")
                self.ax.set_ylabel("ylabel")
                self.ax.legend()
                self.draw()
        elif trend_type.currentText() == 'Параболический':
            if a_.text().replace("-","").isnumeric() and b_.text().replace("-","").isnumeric() and c_.text().replace("-","").isnumeric():
                a = float(a_.text())
                b = float(b_.text())
                c = float(c_.text()) 
                if sposob_sv.currentText() == 'None':
                    y = a + b*t + c*t**2
                elif sposob_sv.currentText() == 'Additive':
                    y = (a + b*t+e + c*t**2)*opred + e*sluch
                elif sposob_sv.currentText() == 'Multiplicative':
                    y = (a + b*t + c*t**2)*e
                self.ax.plot(t, y, marker='<', color='indigo', linewidth=2, alpha=0.4, label="Параболический")  # 
                self.ax.set_title("Title", fontsize=12, fontweight=0)
                self.ax.set_xlabel("xlabel")
                self.ax.set_ylabel("ylabel")
                self.ax.legend()
                self.draw()
        elif trend_type.currentText() == 'Гиперболический':
            if a_.text().replace("-","").isnumeric() and b_.text().replace("-","").isnumeric():
                a = float(a_.text())
                b = float(b_.text())
                y = a + b/t
                if sposob_sv.currentText() == 'Additive':
                    y = y*opred + e*sluch
                elif sposob_sv.currentText() == 'Multiplicative':
                    y = y * e
                self.ax.plot(t, y, marker='v', color='darkcyan', linewidth=2, alpha=0.4, label="Гиперболический")  # 
                self.ax.set_title("Title", fontsize=12, fontweight=0)
                self.ax.set_xlabel("xlabel")
                self.ax.set_ylabel("ylabel")
                self.ax.legend()
                self.draw()
        elif trend_type.currentText() == 'Экспоненциальный':
            if a_.text().replace("-","").isnumeric() and b_.text().replace("-","").isnumeric():
                a = float(a_.text())
                b = float(b_.text())
                y = a*np.exp(b*t)
                if sposob_sv.currentText() == 'Additive':
                    y = y*opred + e*sluch
                elif sposob_sv.currentText() == 'Multiplicative':
                    y = y * e
                self.ax.plot(t, y, marker='^', color='navy', linewidth=2, alpha=0.4, label="Экспоненциальный")  # 
                self.ax.set_title("Title", fontsize=12, fontweight=0)
                self.ax.set_xlabel("xlabel")
                self.ax.set_ylabel("ylabel")
                self.ax.legend()
                self.draw()
        elif trend_type.currentText() == 'Показательный':
            if a_.text().replace("-","").isnumeric() and b_.text().replace("-","").isnumeric():
                a = float(a_.text())
                b = float(b_.text())
                y = a*b**t
                if sposob_sv.currentText() == 'Additive':
                    y = y*opred + e*sluch 
                elif sposob_sv.currentText() == 'Multiplicative':
                    y = y * e
                self.ax.plot(t, y, marker='p', color='c', linewidth=2, alpha=0.4, label="Показательный")  # 
                self.ax.set_title("Title", fontsize=12, fontweight=0)
                self.ax.set_xlabel("xlabel")
                self.ax.set_ylabel("ylabel")
                self.ax.legend()
                self.draw()
        elif trend_type.currentText() == 'Степенной':
            if a_.text().replace("-","").isnumeric() and b_.text().replace("-","").isnumeric():
                a = float(a_.text())
                b = float(b_.text())
                y = a*t**b
                if sposob_sv.currentText() == 'Additive':
                    y = y*opred + e*sluch 
                elif sposob_sv.currentText() == 'Multiplicative':
                    y = y * e
                self.ax.plot(t, y, marker='8', color='gray', linewidth=2, alpha=0.4, label="Степенной")  # 
                self.ax.set_title("Title", fontsize=12, fontweight=0)
                self.ax.set_xlabel("xlabel")
                self.ax.set_ylabel("ylabel")
                self.ax.legend()
                self.draw()
        elif trend_type.currentText() == 'Логарифмический':
            if a_.text().replace("-","").isnumeric() and b_.text().replace("-","").isnumeric():
                a = float(a_.text())
                b = float(b_.text())
                y = a + b*np.log(t)
                if sposob_sv.currentText() == 'Additive':
                    y = y*opred + e*sluch 
                elif sposob_sv.currentText() == 'Multiplicative':
                    y = y * e
                self.ax.plot(t, y, marker='s', color='sandybrown', linewidth=2, alpha=0.4, label="Логарифмический")  # 
                self.ax.set_title("Title", fontsize=12, fontweight=0)
                self.ax.set_xlabel("xlabel")
                self.ax.set_ylabel("ylabel")
                self.ax.legend()
                self.draw()
        elif trend_type.currentText() == 'Гармонический':
            if a_.text().replace("-","").isnumeric() and b_.text().replace("-").isnumeric() and c_.text().replace("-").isnumeric():
                a = float(a_.text())
                b = float(b_.text())
                c = float(c_.text())
                y = a*np.cos(2*np.pi*b*t + c)
                self.ax.plot(t, y, marker='*', color='green', linewidth=1.8, alpha=0.7, label="Гармонический")  # 
                self.ax.set_title("Title", fontsize=12, fontweight=0)
                self.ax.set_xlabel("xlabel")
                self.ax.set_ylabel("ylabel")
                self.ax.legend()
                self.draw()
                
    def plot_loaded(self, data):
        self.clear()
        sns.set_style('darkgrid')
        plt.style.use('ggplot')
        self.ax = self.figure.add_subplot(1,1,1)
        try:
            self.ax.plot(data['t'], data['y'], marker='.', color='black', linewidth=3, alpha=0.5, label="Наблюдения")  # 
            self.ax.set_title("Загруженные данные", fontsize=12, fontweight=0)
            self.ax.set_xlabel("Время, t")
            self.ax.set_ylabel("Значение, y")
            self.ax.legend()
            self.draw()
        except IndexError:
            pass

    def plot_fitted(self,estimator = None, combo = None, data = np.empty(shape=(1,2))):
        plt.style.use('ggplot')
        sns.set_style('darkgrid')
        try:
            estimator
            self.ax = self.figure.add_subplot(1,1,1)
            try:
                self.axes.legend_.texts
            except:
                self.ax.plot(data['t'], data['y'], marker='.', color='black', linewidth=3, alpha=0.5, label="Наблюдения")
                self.ax.legend()
                self.draw()  # 
            if not 'Наблюдения' in self.axes.legend_.texts[0]._text:
                self.ax.plot(data['t'], data['y'], marker='.', color='black', linewidth=3, alpha=0.5, label="Наблюдения")  # 
                self.ax.legend()
                self.draw()
            if isinstance(estimator, LinearRegression) and combo.currentText() =='Линейный':
                data['y_hat'] = estimator.intercept_[0] + estimator.coef_[0][0] * data['t']
                trend_eq = f'y = {round(estimator.intercept_[0],2)} + {round(estimator.coef_[0][0],2)} * x'
                self.ax.plot(data['t'], data['y_hat'], marker='d', linewidth=1.5, alpha=0.75, label="Лин. тренд: " + trend_eq) # 
                # 

            elif isinstance(estimator, LinearRegression) and combo.currentText() =='Логарифмический':
                data['y_hat'] = estimator.intercept_[0] + estimator.coef_[0][0] * np.log(data['t'])
                trend_eq = f'y = {round(estimator.intercept_[0],2)} + {round(estimator.coef_[0][0],2)} * ln(x)'
                self.ax.plot(data['t'], data['y_hat'], marker='X', linewidth=1.5, alpha=0.75, label="Лог. тренд: " + trend_eq) # 
                
            elif isinstance(estimator, LinearRegression) and combo.currentText() =='Параболический':
                data['y_hat'] = estimator.intercept_[0] + estimator.coef_[0][0] * data['t'] + estimator.coef_[-1][-1] * data['t']**2
                trend_eq = f'y = {round(estimator.intercept_[0],2)} + {round(estimator.coef_[0][0],2)} * x + {round(estimator.coef_[-1][-1],2)} * x^2'
                self.ax.plot(data['t'], data['y_hat'], marker='x', linewidth=1.5, alpha=0.75, label="Параб. тренд: " + trend_eq) # 

            elif isinstance(estimator, LinearRegression) and combo.currentText() =='Гиперболический':
                data['y_hat'] = estimator.intercept_[0] + estimator.coef_[0][0] / data['t']
                trend_eq = f'y = {round(estimator.intercept_[0],2)} + {round(estimator.coef_[0][0],2)} / x'
                self.ax.plot(data['t'], data['y_hat'], marker='v', linewidth=1.5, alpha=0.75, label="Гипер. тренд: " + trend_eq)
            elif isinstance(estimator, LinearRegression) and combo.currentText() =='Экспоненциальный':
                #y=a*e^(b*t)
                #ln(y) = ln(a) + b*t
                #Y=A+b*t
                data['y_hat'] = np.exp(estimator.intercept_[0]) * np.exp(estimator.coef_[0][0] * data['t'])
                trend_eq = f'y = {round( np.exp(estimator.intercept_[0]),2)} * e^{round(estimator.coef_[0][0],2)} * x'
                self.ax.plot(data['t'], data['y_hat'], marker='<', linewidth=1.5, alpha=0.75, label="Экспон. тренд: " + trend_eq) # 
            elif isinstance(estimator, LinearRegression) and combo.currentText() =='Показательный':
                #y=a*b^t
                #ln(y) = ln(a) + ln(b) * t
                #Y=A+B*t
                data['y_hat'] = np.exp(estimator.intercept_[0]) * np.exp(estimator.coef_[0][0]) ** data['t']
                trend_eq = f'y = {round(np.exp(estimator.intercept_[0]),2)} * {round(np.exp(estimator.coef_[0][0]),2)} ^ x'
                self.ax.plot(data['t'], data['y_hat'], marker='>', linewidth=1.5, alpha=0.75, label="Показ. тренд: " + trend_eq)
            elif isinstance(estimator, LinearRegression) and combo.currentText() =='Степенной':
                #y=a*t^b
                #ln(y) = ln(a) + b*ln(t)
                #Y=A+b*T
                data['y_hat'] = np.exp(estimator.intercept_[0])* (data['t'] ** estimator.coef_[0][0]) 
                trend_eq = f'y = {round(np.exp(estimator.intercept_[0]),2)} * x ^ {round(estimator.coef_[0][0],2)}'
                self.ax.plot(data['t'], data['y_hat'], marker='.', linewidth=1.5, alpha=0.75, label="Степ. тренд: " + trend_eq) # 
            elif isinstance(estimator, LinearRegression) and combo.currentText() =='Гармонический':
                pass
            self.ax.set_title("Title", fontsize=12, fontweight=0)
            self.ax.set_xlabel("xlabel")
            self.ax.set_ylabel("ylabel")
            self.ax.legend()
            self.draw()
        
        except AttributeError:
            pass

if __name__ == '__main__':

    app = QApplication(sys.argv)
    w = Example()
    sys.exit(app.exec_())

'''
        FigureCanvas.setSizePolicy(self,
                QSizePolicy.Expanding,
                QSizePolicy.Expanding)
'''

