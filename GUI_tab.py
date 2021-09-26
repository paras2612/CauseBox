from PyQt5.QtWidgets import *
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
import matplotlib.pyplot as plt
import os
import json
import csv

table1 = {
    "CFRNET":"0.00",
    "SITE":"0.00",
    "BART":"0.00",
    "CForest":"0.00",
    "DRNET":"0.00",
    "CEVAE":"0.00",
    "PM":"0.00"

}

table2 ={
    "CFRNET":"0.00",
    "SITE":"0.00",
    "BART":"0.00",
    "CForest":"0.00",
    "DRNET":"0.00",
    "CEVAE":"0.00",
    "PM":"0.00"
}
table3 ={
    "CFRNET1":"0.00",
    "CFRNET2":"0.00"
}
table4 ={
    "CFRNET1":"0.00",
    "CFRNET2":"0.00"
}

class FixFigureCanvas(FigureCanvas):
    def resizeEvent(self, event):
        if event.size().width() <= 0 or event.size().height() <= 0:
            return
        super(FixFigureCanvas, self).resizeEvent(event)

class comparisonTab(QTabBar):
    def __init__(self):
        super().__init__()
        self.dataset = "Jobs"
        self.tabWindowGUI()

    def updateResultData(self, modelName, dataset, metric):
        print( modelName, dataset, metric)
        if dataset == "Jobs":
            table1[modelName] = metric
        elif dataset == "IHDP":
            table2[modelName] = metric
        self.updatePlotBox()
        self.updateTableBox()

    def tabWindowGUI(self):
        self.mainLayout = QGridLayout()
        self.dataComboBox = self.combobox()
        self.mainLayout.addWidget(self.dataComboBox, 0, 0)
        #Please let image downloaded with a button!!!
        self.plotBox = self.plotbox()
        self.mainLayout.addWidget(self.plotBox, 1, 0)
        self.tableBox = self.tablebox()
        self.mainLayout.addWidget(self.tableBox, 2, 0)
        # self.resultBox = self.resultbox()
        # self.mainLayout.addWidget(self.resultBox, 3, 0)
        self.setLayout(self.mainLayout)

    def drawPlot2(self):
        figure = plt.figure()
        figure.clf()
        figure.subplots_adjust(hspace=0.5)
        figure.subplots_adjust(wspace=0.5)
        canvas = FixFigureCanvas(figure)

        ax = figure.add_subplot(111)
        x = table1.keys()
        y = [float(value) for value in table2.values()]
        ax.bar(x, y, color=["red", "green", "blue", "black"])
        # ax.set_title("IHDP Dataset")
        ax.set_ylabel('PEHE')
        ax.set_xlabel('Model')
        plt.xticks(rotation=45)

        canvas.draw_idle()
        return canvas

    def drawPlot1(self):
        figure = plt.figure()
        figure.clf()
        figure.subplots_adjust(hspace=0.5)
        figure.subplots_adjust(wspace=0.5)
        canvas = FixFigureCanvas(figure)

        ax = figure.add_subplot(111)
        x = table2.keys()
        y = [float(value) for value in table1.values()]
        ax.bar(x, y, color=["red", "green", "blue", "black"])
        # ax.set_title("Jobs Dataset")
        ax.set_ylabel('Policy Risk')
        ax.set_xlabel('Model')
        plt.xticks(rotation=45)
        canvas.draw_idle()
        return canvas

    def plotbox(self):
        if self.dataset == "Jobs":
            canvas = self.drawPlot1()
        elif self.dataset == "IHDP":
            canvas = self.drawPlot2()
        else:
            print("no such dataset available")
        return canvas

    def drawTable1(self):
        numrow = 7
        numcol = 2
        widget = QTableWidget(numrow, numcol)
        widget.setEditTriggers(QTableWidget.NoEditTriggers)
        widget.setSizeAdjustPolicy(QAbstractScrollArea.AdjustToContents)
        widget.setHorizontalHeaderLabels(
            ['Model', 'Policy Risk']
        )
        widget.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        i = j = 0
        for key, value in table1.items():
            widget.setItem(i, j, QTableWidgetItem(str(key)))
            widget.setItem(i, j+1, QTableWidgetItem(str(value)))
            i+=1
            j=0
        return widget

    def drawTable2(self):
        numrow = 7
        numcol = 2
        widget = QTableWidget(numrow, numcol)
        widget.setEditTriggers(QTableWidget.NoEditTriggers)
        widget.setSizeAdjustPolicy(QAbstractScrollArea.AdjustToContents)
        widget.setHorizontalHeaderLabels(
            ['Model', 'PEHE']
        )
        widget.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        i = j = 0
        for key, value in table2.items():
            widget.setItem(i, j, QTableWidgetItem(str(key)))
            widget.setItem(i, j+1, QTableWidgetItem(str(value)))
            i+=1
            j=0
        return widget

    def tablebox(self):
        if self.dataset == "Jobs":
            self.tableBox = self.drawTable1()
        elif self.dataset == "IHDP":
            self.tableBox = self.drawTable2()
        else:
            print("no such dataset available")
        return self.tableBox

    def resultbox(self):
        widget = QWidget()
        layout = QVBoxLayout()
        plotBox = self.plotbox()
        layout.addWidget(plotBox)

        tableBox = self.tablebox()
        layout.addWidget(tableBox)
        widget.setLayout(layout)
        return widget

    def combobox(self):
        self.dataComboBox = QComboBox()
        self.dataComboBox.addItem("Jobs")
        self.dataComboBox.addItem("IHDP")
        self.dataComboBox.activated[str].connect(self.dataChoice)
        return self.dataComboBox

    def dataChoice(self, text):
        self.dataset = text
        self.updatePlotBox()
        self.updateTableBox()

    def updatePlotBox(self):
        if hasattr(self, 'plotBox'):
            self.mainLayout.removeWidget(self.plotBox)
            self.plotBox = self.plotbox()
            self.mainLayout.addWidget(self.plotBox, 1, 0)
            self.mainLayout.update()

    def updateTableBox(self):
        if hasattr(self, 'tableBox'):
            self.mainLayout.removeWidget(self.tableBox)
            self.plotBox = self.tablebox()
            self.mainLayout.addWidget(self.tableBox, 2, 0)
            self.mainLayout.update()

    def updateResultBox(self):
        if hasattr(self, 'resultBox'):
            self.mainLayout.removeWidget(self.resultBox)
            self.plotBox = self.resultbox()
            self.mainLayout.addWidget(self.resultBox, 1, 0)
            self.mainLayout.update()

class hyperparamsTab(QTabBar):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout()
        self.combobox()
        layout.addWidget(self.modelComboBox)
        self.figure = plt.figure()
        self.canvas = FigureCanvas(self.figure)
        self.plot()
        layout.addWidget(self.canvas)

        self.table()
        layout.addWidget(self.tableWidget)

        self.setLayout(layout)
    def combobox(self):
        self.modelComboBox = QComboBox()
        self.modelComboBox.addItem("Counterfactual Regression Network (CRFNet)")
        self.modelComboBox.addItem("Causal Effect Inference with Deep Latent-Variable Models (CEVAE)")
        self.modelComboBox.addItem("Bayesian Additive Regression Trees (BART)")
        self.modelComboBox.addItem("Causal Forests")
        self.modelComboBox.addItem("Perfect Match")
        self.modelComboBox.addItem("Learning Disentangled Representations for counterfactual regression (DRNet)")
        self.modelComboBox.addItem("Local similarity preserved individual treatment effect (SITE)")
        # self.modelComboBox.activated[str].connect(self.modelChoice)

    def table(self):
        self.numrow = 50
        numcol = 50
        self.tableWidget = QTableWidget(self.numrow, numcol)
        #Need to change dynamicall refelec attrivtues
        with open("ParamsInputCFR", "r") as file:
            text  = file.read()
        headers = [param.split()[0] for param in text.split("\n")]
        headers.insert(0, "Model")
        self.tableWidget.setHorizontalHeaderLabels(
            headers
        )
        self.modelbuttons = list()
        dirpath = os.getcwd() + "\\PlotData\\"
        files = os.listdir(dirpath)
        temp = ["CFRNET1", "CFRNET2", "CFRNET3"]
        for row, file in enumerate(files):
            with open(dirpath+file) as result:
                data = json.load(result)
            if row == 0:
                selectplotbtn = QPushButton("Select"+" "+temp[0])
            elif row == 1:
                selectplotbtn = QPushButton("Select"+" "+temp[1])

            self.tableWidget.setCellWidget(row, 0, selectplotbtn)
            # self.tableWidget.setEditTriggers(QTableWidget.NoEditTriggers)
            col = 0
            for key, val in data["params"].items():
                self.tableWidget.setItem(row, col, QTableWidgetItem(str(val)))
                col+=1
            self.modelbuttons.append(selectplotbtn)

    def plot(self):
        self.figure.clf()
        self.figure.subplots_adjust(hspace=0.5)
        self.figure.subplots_adjust(wspace=0.5)

        ax1 = self.figure.add_subplot(121)
        x1 = table3.keys()
        y1 = [float(value) for value in table3.values()]
        ax1.bar(x1, y1, color='rgbc')
        ax1.set_title("IHDP Dataset")
        ax1.set_ylabel('PEHE')
        ax1.set_xlabel('Model')
        # plt.xticks(rotation=45)

        ax2 = self.figure.add_subplot(122)
        x2 = table4.keys()
        y2 = [float(value) for value in table4.values()]
        ax2.bar(x2, y2, color='rgbc')
        ax2.set_title("IHDP Dataset")
        ax2.set_ylabel('Error on ATE')
        ax2.set_xlabel('Model')
        # plt.xticks(rotation=45)

        self.canvas.draw_idle()