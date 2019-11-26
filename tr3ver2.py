import sys
from PyQt5.QtWidgets import (QMainWindow, QAction, qApp, QApplication, QPushButton, QDesktopWidget,
                            QLabel,QRadioButton, QFileDialog, QWidget, QGridLayout, QMenu, QSizePolicy, QMessageBox, QWidget)
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
from scipy.io.wavfile import read

import matplotlib.pyplot as plt
import numpy as np
import wave
import sys

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
        self.setWindowTitle("Combobox example") 
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
        plotCan = PlotCanvas(self, width=7, height=4)
        #grid.addWidget(plotCan, 0,1)
        plotCan.move(250,0)

        

        #button
        #btn = QPushButton("Load Data", self)
        #btn.move(100,200)
        #btn.setVisible(True)
        #btn.resize(btn.sizeHint())
        #grid.addWidget(btn, 1,0)
        #btn.clicked.connect(lambda: plotCan.plot(self.data)) #good
        
        

        #label
        self.eq_label = QLabel(self)
        self.eq_label.move(8,30)
        #self.eq_label.setText('Fit')
        self.eq_label.setGeometry(QtCore.QRect(25, 0, 500, 100)) #(x, y, width, height)


        #combobox
        self.combo = QComboBox(self)
        self.combo.move(8,70)
        self.combo.resize(150,25)
        self.combo.addItems(['Линейная','Логарифмическая'])
        self.combo.activated[str].connect(self.combobox_changed)

        self.btn2 = QPushButton('Fit Regression',self)
        self.btn2.move(8, 230)
        self.btn2.resize(150,30)

        self.btn2.clicked.connect(self.fit_regression)
        #self.btn2.clicked.connect(lambda: plotCan.plot(self.data))
        self.btn2.clicked.connect(lambda: plotCan.plot_fitted(self.estimator, self.combo, self.data))     #plot fitted

        btn3 = QPushButton('Open file', self)
        btn3.move(8, 170)
        btn3.resize(150,30)
        btn3.clicked.connect(self.openFile)
        #btn3.clicked.connect(lambda: plotCan.plot(self.data))

        btn4 = QPushButton('Clear', self)
        btn4.move(8, 200)
        btn4.resize(150,30)
        btn4.clicked.connect(lambda: plotCan.clear(self))

        self.show()

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
            filter="CSV files (*.csv);;Excel files (*.xlsx);;""WAV files (*.wav);;Binary files (*.bin)")[0]
            if self.filepath.endswith('csv'):
                self.data = pd.read_csv(self.filepath, delimiter=',', header=None)
                print('csv')
                print(type(self.data))
            elif self.filepath.endswith('xlsx'):
                self.data = pd.read_excel(self.filepath, header=None)
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
                self.data = df[df[0] != 0]
                print('wav')
                print(self.data.shape)
        except OSError:
            pass

    def buttonpress(self):
        self.plot(self.data)

    def combobox_changed(self, text):
        if text == 'Линейная':
            self.eq_label.setText("y=a+b*x")
            self.btn2.setText('Fit Linear Regression')
        elif text == 'Логарифмическая':
            self.eq_label.setText("y=a+b*ln(x)")
            self.btn2.setText('Fit Log Regression')
        
        print(text)

    def fit_regression(self):
        try:
            if self.combo.currentText() =='Линейная':
                self.estimator = LinearRegression()
                fitted_estimator = self.estimator.fit(X = self.data.loc[:, 0].values.reshape(-1,1), y = self.data.loc[:, 1].values.reshape(-1,1))
                self.equation = f'y = {round(fitted_estimator.intercept_[0],2)} + {round(fitted_estimator.coef_[0][0],2)} * x;' + '\n' + f'R-squared = {round(fitted_estimator.score(X = self.data.loc[:, 0].values.reshape(-1,1), y = self.data.loc[:, 1].values.reshape(-1,1)),2)}'
                self.eq_label.setText(self.equation)
            elif self.combo.currentText() =='Логарифмическая':
                self.estimator = LinearRegression()
                fitted_estimator = self.estimator.fit(X = np.log(self.data.loc[:, 0].values.reshape(-1,1)), y = self.data.loc[:, 1].values.reshape(-1,1))
                self.equation = (f'y = {round(fitted_estimator.intercept_[0],2)} + {round(fitted_estimator.coef_[0][0],2)} * ln(x);' + '\n' + f'R-squared = {round(fitted_estimator.score(X = self.data.loc[:, 0].values.reshape(-1,1), y = self.data.loc[:, 1].values.reshape(-1,1)),2)}')
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
        self.plot()

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
        ax = self.figure.add_subplot(111)
        try:
            ax.plot(data.loc[:,0],data.loc[:,1], 'r-')
        except:
            ax.plot(data[:,0],data[:,1], 'r-')
        ax.set_title('PyQt Matplotlib Example')
        self.draw()

    def plot_fitted(self,estimator = None, combo = None, data = np.empty(shape=(1,2))):
        sns.set_style('darkgrid')
        try:
            estimator
            self.ax = self.figure.add_subplot(1,1,1)
            if isinstance(estimator, LinearRegression) and combo.currentText() =='Линейная':
                data['y_hat'] = estimator.intercept_[0] + estimator.coef_[0][0] * data[0]
                trend_eq = f'y = {round(estimator.intercept_[0],2)} + {round(estimator.coef_[0][0],2)} * x'
                self.ax.plot(data[0], data['y_hat'], marker='', linewidth=1, alpha=0.7, label="Лин. тренд: " + trend_eq) # 
                self.ax.plot(data[0], data[1], marker='', color='orange', linewidth=1, alpha=0.7, label="obs")  # 
                self.ax.set_title("Title", fontsize=12, fontweight=0)
                self.ax.set_xlabel("xlabel")
                self.ax.set_ylabel("ylabel")
                self.ax.legend()
                self.draw()
            elif isinstance(estimator, LinearRegression) and combo.currentText() =='Логарифмическая':
                data['y_hat'] = estimator.intercept_[0] + estimator.coef_[0][0] * np.log(data[0])
                trend_eq = f'y = {round(estimator.intercept_[0],2)} + {round(estimator.coef_[0][0],2)} * ln(x)'
                self.ax.plot(data[0], data['y_hat'], marker='', linewidth=1, alpha=0.7, label="Лог. тренд: " + trend_eq) # 
                self.ax.plot(data[0], data[1], marker='', color='orange', linewidth=1, alpha=0.7, label="obs")  # 
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
