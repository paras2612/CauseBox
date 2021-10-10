import itertools

from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt
import threadModel
import os
import csv


class FixFigureCanvas(FigureCanvas):
    def resizeEvent(self, event):
        if event.size().width() <= 0 or event.size().height() <= 0:
            return
        super(FixFigureCanvas, self).resizeEvent(event)


class comparisonTab(QTabBar):
    def __init__(self):
        super().__init__()
        self.table1 = {
            "CFRNET": "0.00",
            "SITE": "0.00",
            "BART": "0.00",
            "CForest": "0.00",
            "DRNET": "0.00",
            "CEVAE": "0.00",
            "PM": "0.00"
        }

        self.table2 = {
            "CFRNET": "0.00",
            "SITE": "0.00",
            "BART": "0.00",
            "CForest": "0.00",
            "DRNET": "0.00",
            "CEVAE": "0.00",
            "PM": "0.00"
        }
        self.dataset = "Jobs"
        self.tabWindowGUI()

    def tabWindowGUI(self):
        self.mainLayout = QGridLayout()
        self.dataComboBox = self.combobox()
        self.mainLayout.addWidget(self.dataComboBox, 0, 0)
        # Please let image downloaded with a button!!!
        self.plotBox = self.plotbox()
        self.mainLayout.addWidget(self.plotBox, 1, 0)
        self.tableBox = self.tablebox()
        self.mainLayout.addWidget(self.tableBox, 2, 0)
        self.setLayout(self.mainLayout)

    def drawPlot2(self):
        figure = plt.figure()
        figure.clf()
        figure.subplots_adjust(hspace=0.5)
        figure.subplots_adjust(wspace=0.5)
        canvas = FixFigureCanvas(figure)

        ax = figure.add_subplot(111)
        x = self.table1.keys()
        y = [float(value) for value in self.table2.values()]

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
        x = self.table2.keys()
        y = [float(value) for value in self.table1.values()]
        ax.bar(x, y, color=["red", "green", "blue", "black"])
        # ax.set_title("Jobs Dataset")
        ax.set_ylabel('Policy Risk')
        ax.set_xlabel('Model')
        plt.xticks(rotation=45)
        canvas.draw_idle()
        return canvas

    def plotbox(self):
        if self.dataset.lower() == "jobs":
            canvas = self.drawPlot1()
        elif self.dataset.lower() == "ihdp":
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
        for key, value in self.table1.items():
            widget.setItem(i, j, QTableWidgetItem(str(key)))
            #FUTURE: Change value 0.00 with N/A using filter(lambda x: x % 2 != 0, seq)
            widget.setItem(i, j + 1, QTableWidgetItem(str(value)))
            i += 1
            j = 0
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
        for key, value in self.table2.items():
            widget.setItem(i, j, QTableWidgetItem(str(key)))
            widget.setItem(i, j + 1, QTableWidgetItem(str(value)))
            i += 1
            j = 0
        return widget

    def tablebox(self):
        if self.dataset.lower() == "jobs":
            self.tableBox = self.drawTable1()
        elif self.dataset.lower() == "ihdp":
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

    def updateResultData(self, modelName, dataset, metric):
        if dataset.lower() == "jobs":
            self.table1[modelName] = metric
        elif dataset.lower() == "ihdp":
            self.table2[modelName] = metric
        self.updatePlotBox()
        self.updateTableBox()


class hyperparamsTab(QTabBar):
    def __init__(self):
        super().__init__()
        self.config_path = ""
        self.hyper_dicts = dict()
        self.num_models = len(self.hyper_dicts)
        self.table1 = dict()
        self.table2 = dict()
        self.dataset = "Jobs"
        self.modelName = "Counterfactual Regression Network (CFRNet)"
        self.tabWindowGUI()

    def setConfigLocation(self, filename):
        self.config_path = filename
        self.hyper_dicts = self.createOptions()
        self.updateTableBox()

    def tabWindowGUI(self):
        self.mainLayout = QGridLayout()
        self.modelComboBox = self.combobox()
        self.mainLayout.addWidget(self.modelComboBox, 0, 0)
        # Please let image downloaded with a button!!!
        self.plotBox = self.plotbox()
        self.mainLayout.addWidget(self.plotBox, 1, 0)
        self.tableBox = self.tablebox()
        self.mainLayout.addWidget(self.tableBox, 2, 0)
        self.setLayout(self.mainLayout)

    def loadParamserchConfig(self):
        result = dict()

        with open(self.config_path, "r") as f:
            text = "".join(f.readlines())
            delimParamsList = [option.split("=") for option in text.split("\n")]
            for key, val in delimParamsList:
                key = key.lower()
                result[key] = val.strip('\"')
        return result

    def unfoldHyperparmSearch(self, hyperParams):
        variation1 = hyperParams["p_alpha"].strip('][').split(',')
        variation1 = [float(value) for value in variation1]
        variation2 = hyperParams["p_lambda"].strip('][').split(',')
        variation2 = [float(value) for value in variation2]
        del hyperParams["p_alpha"]
        del hyperParams["p_lambda"]

        options = list()
        combinations = list(itertools.product(variation1, variation2))
        for combination in combinations:
            option = hyperParams.copy()
            option["p_alpha"] = combination[0]
            option["p_lambda"] = combination[1]
            options.append(option)
        return options

    def createOptions(self):
        configParams = self.loadParamserchConfig()
        result = self.unfoldHyperparmSearch(configParams)
        return result

    def createCommand(self, hyper_dict):
        base_command = "python main.py cfrnet"
        options = list()
        for key, val in hyper_dict.items():
            key = key.lower()
            option = "--{key} {val}".format(key=key, val=val)
            options.append(option)
        result = base_command + " ".join(options)
        return result

    def threadHyperparamModel(self, command, dataset, modelName, experiments):
        self.objThread = QThread()
        print("****************************")
        print(dataset, modelName, experiments)
        self.obj = threadModel.backgroundApp(command, dataset, modelName, experiments)
        self.obj.moveToThread(self.objThread)
        self.obj.finished.connect(self.objThread.quit)
        self.objThread.started.connect(self.obj.run)
        self.objThread.finished.connect(self.updateResult)
        self.objThread.start()

    def tabWindowGUI(self):
        self.mainLayout = QGridLayout()
        self.dataComboBox = self.combobox()
        self.mainLayout.addWidget(self.dataComboBox, 0, 0)
        # Please let image downloaded with a button!!!
        self.plotBox = self.plotbox()
        self.mainLayout.addWidget(self.plotBox, 1, 0)
        self.tableBox = self.tablebox()
        self.mainLayout.addWidget(self.tableBox, 2, 0)
        self.setLayout(self.mainLayout)

    def drawPlot1(self):
        figure = plt.figure()
        figure.clf()
        figure.subplots_adjust(hspace=0.5)
        figure.subplots_adjust(wspace=0.5)
        canvas = FixFigureCanvas(figure)
        ax1 = figure.add_subplot(111)
        x1 = list(self.table1.keys())
        y1 = [float(value) for value in self.table1.values()]
        ax1.bar(x1, y1)#, color='rgbc')
        ax1.set_ylabel('Policy Risk')
        ax1.set_xlabel('Run Names')
        plt.xticks(rotation=45)
        return canvas

    def drawPlot2(self):
        figure = plt.figure()
        figure.clf()
        figure.subplots_adjust(hspace=0.5)
        figure.subplots_adjust(wspace=0.5)
        canvas = FixFigureCanvas(figure)

        ax2 = figure.add_subplot(111)
        x2 = list(self.table2.keys())
        y2 = [float(value) for value in self.table2.values()]
        ax2.bar(x2, y2)#, color='rgbc')
        # ax2.set_title("Jobs Dataset")
        ax2.set_ylabel('PEHE')
        ax2.set_xlabel('Run Names')
        plt.xticks(rotation=45)

        canvas.draw_idle()
        return canvas

    def plotbox(self):
        if self.dataset.lower() == "jobs":
            canvas = self.drawPlot1()
        elif self.dataset.lower() == "ihdp":
            canvas = self.drawPlot2()
        return canvas

    def tablebox(self):
        if self.hyper_dicts == {}:
            widget = QTableWidget(0,0)
            return widget

        headers = list(self.hyper_dicts[0].keys())
        headers.insert(0, "Click")
        numrow = len(self.hyper_dicts)
        numcol = len(headers)
        widget = QTableWidget(numrow, numcol)

        widget.setHorizontalHeaderLabels(
            headers
        )
        self.hyperparamBtns = list()
        row_idx = 0
        for hyper_dict in self.hyper_dicts:
            hyperparamsBtn = QPushButton("Run " + str(row_idx))
            hyperparamsBtn.clicked.connect(self.experimentChoice)
            widget.setCellWidget(row_idx, 0, hyperparamsBtn)
            col_idx = 1
            for key, val in hyper_dict.items():
                widget.setItem(row_idx, col_idx, QTableWidgetItem(str(val)))
                col_idx += 1
            self.hyperparamBtns.append(hyperparamsBtn)
            row_idx += 1
        return widget

    def updateResult(self):
        try:
            # for demo....
            # filename = "Results.csv"
            filename = "tempResult.csv"
            rows = self.readResultCSV(filename)
            formatter = "{0:.2f}"
            # Would be better to change it to pandas dataframe and process
            for row in rows:
                # modelName = row[0].upper().strip()
                dataset = row[1].strip()
                if dataset.lower() == "jobs":
                    metric = float(row[-1].strip())  # Policy Risk
                elif dataset.lower() == "ihdp":
                    metric = float(row[-1].strip())  # PEHE
                    #change it to -3 next time... demo ...
                else:
                    print("no such metric exist in the result file")
                metric = formatter.format(metric)
        except:
            print("Couldn't find the result file")
        self.updateResultData(self.runName, dataset, metric)

    def updateResultData(self, modelName, dataset, metric):
        if dataset.lower() == "jobs":
            self.table1[modelName] = metric
        elif dataset.lower() == "ihdp":
            self.table2[modelName] = metric
        self.updatePlotBox()

    def readResultCSV(self, filename):
        filepath = os.getcwd() + "\\" + filename
        # filepath = filename
        fields = []
        rows = list()
        with open(filepath, 'r') as csvfile:
            # creating a csv reader object
            csvreader = csv.reader(csvfile)

            # extracting field names through first row
            # fields = next(csvreader)

            # extracting each data row one by one
            for row in csvreader:
                rows.append(row)
            # get total number of rows
        return rows

    def experimentChoice(self):
        text = self.sender().text()
        self.runName = text#Used in csv reading and append to the table
        exp_idx = int(text.split()[1])
        options = self.hyper_dicts[exp_idx]
        dataset = options["dataset"]
        experiments = options["experiments"]

        modelName = "Counterfactual Regression Network (CFRNet)"
        command = self.createCommand(options)
        self.threadHyperparamModel(command, dataset, modelName, experiments)

    def combobox(self):
        widget = QWidget()
        layout = QVBoxLayout()
        self.modelComboBox = QComboBox()
        self.modelComboBox.addItem("Counterfactual Regression Network (CFRNet)")
        self.modelComboBox.addItem("Causal Effect Inference with Deep Latent-Variable Models (CEVAE)")
        self.modelComboBox.addItem("Bayesian Additive Regression Trees (BART)")
        self.modelComboBox.addItem("Causal Forests")
        self.modelComboBox.addItem("Perfect Match")
        self.modelComboBox.addItem("Learning Disentangled Representations for counterfactual regression (DRNet)")
        self.modelComboBox.addItem("Local similarity preserved individual treatment effect (SITE)")
        self.modelComboBox.activated[str].connect(self.modelChoice)
        layout.addWidget(self.modelComboBox)
        self.dataComboBox = QComboBox()
        self.dataComboBox.addItem("Jobs")
        self.dataComboBox.addItem("IHDP")
        self.dataComboBox.activated[str].connect(self.dataChoice)
        layout.addWidget((self.dataComboBox))
        widget.setLayout(layout)
        return widget

    def modelChoice(self, text):
        self.modelName = text
        self.updatePlotBox()
        self.updateTableBox()

    def dataChoice(self, text):
        self.dataset = text
        self.updatePlotBox()

    def updatePlotBox(self):
        if hasattr(self, 'plotBox'):
            self.mainLayout.removeWidget(self.plotBox)
            self.plotBox = self.plotbox()
            self.mainLayout.addWidget(self.plotBox, 1, 0)
            self.mainLayout.update()

    def updateTableBox(self):
        if hasattr(self, 'tableBox'):
            self.mainLayout.removeWidget(self.tableBox)
            self.tableBox = self.tablebox()
            self.mainLayout.addWidget(self.tableBox, 2, 0)
            self.mainLayout.update()