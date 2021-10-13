import sys
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
import threadModel
import args
from queue import Queue
from GUI_tab import comparisonTab, hyperparamsTab
import json
import os
import datetime
import csv


class WriteStream(object):
    def __init__(self, queue):
        self.queue = queue

    def write(self, text):
        self.queue.put(text)

    def flush(self):
        pass


class MyReceiver(QObject):
    mysignal = pyqtSignal(str)

    def __init__(self, queue, *args, **kwargs):
        QObject.__init__(self, *args, **kwargs)
        self.queue = queue

    @pyqtSlot()
    def run(self):
        while True:
            text = self.queue.get()
            self.mysignal.emit(text)


class MyApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.args = args.CFRNet()
        self.dataset = "Jobs"
        self.p = None  # Default empty value.
        self.modelName = "Counterfactual Regression Network (CFRNet)"
        self.modelAlias = "cfrnet"
        self.MainWindowGUI()

    def MainWindowGUI(self):
        exitApp = QAction('&Close Window', self)
        exitApp.setShortcut('Ctrl+Q')
        exitApp.setStatusTip('leave the app')
        exitApp.triggered.connect(self.close_application)

        openConfig1 = QAction('&Open Custom Param Config', self)
        openConfig1.setShortcut('Ctrl+1')
        openConfig1.setStatusTip('Open Config For Customized Parameter')
        openConfig1.triggered.connect(self.file_open1)

        openConfig2 = QAction('&Open Param Search Config', self)
        openConfig2.setShortcut('Ctrl+2')
        openConfig2.setStatusTip('Open Config For Hyperparameter Search')
        openConfig2.triggered.connect(self.file_open2)

        fileMenu = self.menuBar().addMenu('&File')
        fileMenu.addAction(openConfig1)
        fileMenu.addAction(openConfig2)
        fileMenu.addAction(exitApp)

        self.SettingsBox = self.createSettingsBox()
        self.ResultTabs = self.createResultTabs()
        self.ParamsBox = self.createParamsBox()
        self.ConsoleBox = self.createConsoleBox()
        self.ButtonBox = self.createButtonBox()

        self.mainLayout = QGridLayout()
        self.mainLayout.addWidget(self.SettingsBox, 0, 0)
        self.mainLayout.addWidget(self.ResultTabs, 0, 1, 3, 1)
        self.mainLayout.addWidget(self.ParamsBox, 1, 0)
        self.mainLayout.addWidget(self.ConsoleBox, 2, 0)
        self.mainLayout.addWidget(self.ButtonBox, 3, 0)

        self.mainWidget = QWidget()
        self.setCentralWidget(self.mainWidget)
        self.mainWidget.setLayout(self.mainLayout)

        self.setWindowIcon(self.style().standardIcon(getattr(QStyle, 'SP_FileDialogListView')))
        self.setWindowTitle('CIKM Demo Tutorial')
        self.setGeometry(2000, 2000, 2000, 2000)
        self.setCenter()

    def createButtonBox(self):
        widget = QWidget()
        layout = QHBoxLayout()

        self.runButton = QToolButton()
        self.runButton.setText("Run The Model With Parameter √")
        self.runButton.clicked.connect(self.runModel)

        self.updateButton = QToolButton()
        self.updateButton.setText("Update The Results To DB √")
        self.updateButton.setEnabled(False)
        self.updateButton.clicked.connect(self.updateResult)

        layout.addWidget(self.runButton)
        layout.addWidget(self.updateButton)
        widget.setLayout(layout)
        return widget

    def createParamsBox(self):
        widget = QWidget()
        layout = QVBoxLayout()

        self.paramLabels = list()
        self.paramLines = list()
        self.RequiredParamsBox = self.createRequiredParamsBox()
        layout.addWidget(self.RequiredParamsBox)

        self.detailsButton = QToolButton()
        self.detailsButton.setText("Less...")
        self.detailsButton.setCheckable(True)
        self.detailsButton.setChecked(True)
        self.detailsButton.setArrowType(Qt.UpArrow)
        self.detailsButton.setAutoRaise(True)
        self.detailsButton.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.detailsButton.clicked.connect(self.showDetail)
        layout.addWidget(self.detailsButton)

        self.OptionalParamsBox = self.createOptionalParamsBox()
        layout.addWidget(self.OptionalParamsBox)

        resetParamsBtn = QPushButton("Reset Params")
        resetParamsBtn.clicked.connect(self.resetParams)
        layout.addWidget(resetParamsBtn)

        widget.setLayout(layout)

        return widget

    def createRequiredParamsBox(self):
        widget = QGroupBox("• Required Parameters: ")
        layout = QGridLayout()
        i = j = 0
        # pairL= QFormLayout()

        for key in self.args.required:
            j += 2
            label = QLabel(key)
            line = QLineEdit()
            # pairL.addRow(label, line)
            layout.addWidget(label, i, j)
            layout.addWidget(line, i, j + 1)
            self.paramLabels.append(label)
            self.paramLines.append(line)

            # Move to next column if the rows are more than 4 lines
            if j > 6:
                i += 1
                j = 0

        widget.setLayout(layout)
        return widget

    def createOptionalParamsBox(self):
        widget = QGroupBox("► Optional Parameters")
        layout = QGridLayout()

        i = j = 0
        for key in self.args.optional:
            j += 2
            label = QLabel(key)
            line = QLineEdit()
            layout.addWidget(label, i, j)
            layout.addWidget(line, i, j + 1)
            self.paramLabels.append(label)
            self.paramLines.append(line)

            # Move to next column if the rows are more than 4 lines
            if j > 6:
                i += 1
                j = 0

        widget.setLayout(layout)
        return widget

    def createConsoleBox(self):
        widget = QGroupBox("Console")
        self.textedit = QTextEdit()

        self.ConsoleLayout = QVBoxLayout()
        self.ConsoleLayout.addWidget(self.textedit)

        clearConsoleBtn = QPushButton("Clear Console")
        clearConsoleBtn.clicked.connect(self.clearConsole)
        self.ConsoleLayout.addWidget(clearConsoleBtn)
        widget.setLayout(self.ConsoleLayout)
        return widget

    def createSettingsBox(self):
        widget = QGroupBox("Settings")

        modelComboBox = QComboBox()
        modelComboBox.addItem("Counterfactual Regression Network (CFRNet)")
        modelComboBox.addItem("Causal Effect Inference with Deep Latent-Variable Models (CEVAE)")
        modelComboBox.addItem("Bayesian Additive Regression Trees (BART)")
        modelComboBox.addItem("Causal Forests")
        modelComboBox.addItem("Perfect Match")
        modelComboBox.addItem("Learning Disentangled Representations for counterfactual regression (DRNet)")
        modelComboBox.addItem("Local similarity preserved individual treatment effect (SITE)")
        modelComboBox.activated[str].connect(self.modelChoice)

        # dataComboBox = QComboBox()
        # dataComboBox.addItem("Jobs")
        # dataComboBox.addItem("IHDP")
        # dataComboBox.activated[str].connect(self.dataChoice)

        self.SettingLayout = QVBoxLayout()
        self.SettingLayout.addWidget(modelComboBox)
        # self.SettingLayout.addWidget(dataComboBox)

        widget.setLayout(self.SettingLayout)
        return widget

    def modelChoice(self, text):
        self.modelName = text
        if self.modelName == "Counterfactual Regression Network (CFRNet)":
            self.args = args.CFRNet()
            self.modelAlias = "cfrnet"
        elif self.modelName == "Causal Effect Inference with Deep Latent-Variable Models (CEVAE)":
            self.args = args.CEVAE()
            self.modelAlias = "cevae"
        elif self.modelName == "Bayesian Additive Regression Trees (BART)":
            self.args = args.BART()
            self.modelAlias = "bart"
        elif self.modelName == "Causal Forests":
            self.args = args.CausalForests()
            self.modelAlias = "cforest"

        elif self.modelName == "Perfect Match":
            self.args = args.PerfectMatch()

        elif self.modelName == "Learning Disentangled Representations for counterfactual regression (DRNet)":
            self.args = args.DRNet()
            self.modelAlias = "drnet"
        elif self.modelName == "Local similarity preserved individual treatment effect (SITE)":
            self.args = args.SITE()
            self.modelAlias = "site"
        else:
            print("No such model available in the combobox setting")

        self.updateWidget()

    def updateWidget(self):
        # self.updateSettingsWidget()
        self.updateParamsWidget()
        # self.updateConsoleWidget()

    def updateSettingsWidget(self):
        if hasattr(self, 'SettingsBox'):
            self.mainLayout.removeWidget(self.SettingsBox)
            self.SettingsBox = self.createSettingsBox()
            self.mainLayout.addWidget(self.SettingsBox, 0, 0)
            self.mainLayout.update()

    def updateParamsWidget(self):
        if hasattr(self, 'ParamsBox'):
            self.mainLayout.removeWidget(self.ParamsBox)
            self.ParamsBox = self.createParamsBox()

            self.mainLayout.addWidget(self.ParamsBox, 1, 0)
            self.mainLayout.update()

    def updateConsoleWidget(self):
        if hasattr(self, 'ConsoleBox'):
            self.mainLayout.removeWidget(self.ConsoleBox)
            self.ConsoleBox = self.createConsoleBox()
            self.mainLayout.addWidget(self.ConsoleBox, 2, 0)
            self.mainLayout.update()

    def dataChoice(self, text):
        self.dataset = text

    def resetParams(self):
        for line in self.paramLines:
            line.setText("")

    def clearConsole(self):
        self.textedit.setText("")

    def runModel(self):
        self.changeBtnStatus()
        self.createCommand()
        # self.start_thread()
        self.start_process()

    def updateResult(self):
        try:
            filename = os.getcwd()+"\\Results.csv"
            rows = self.readResultCSV(filename)
            formatter = "{0:.2f}"
            # Would be better to change it to pandas dataframe and process
            for row in rows:
                modelName = row[0].upper().strip()
                dataset = row[1].strip()
                if dataset.lower() == "jobs":
                    metric = float(row[-1].strip())  # Policy Risk
                elif dataset.lower() == "ihdp":
                    metric = float(row[-3].strip())  # PEHE
                else:
                    print("no such metric exist in the result file")
                metric = formatter.format(metric)
            self.tab1.updateResultData(modelName, dataset, metric)
        except:
            print("Couldn't find the result file")
        self.changeBtnStatus()

    def readResultCSV(self, filename):
        filepath = filename
        print(filepath)
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

    def updateStatus(self):
        self.changeBtnStatus()
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S-%f")

        filename = "result_" + timestamp + ".json"
        filepath = os.getcwd() + "\\PlotData\\" + filename
        result = dict()
        result["evaluation"] = self.extractEvaluation()
        result["params"] = self.newParamDict
        with open(filepath, "w") as file:
            json.dump(result, file, indent=3)

    def extractEvaluation(self):
        for line in self.textedit.toPlainText().split("\n"):
            if "Policy Risk" in line or "ATT" in line:
                evaluation = line
        PolicyRisk = evaluation.split("ATT")[0].split()[3]
        ATT = evaluation.split("ATT")[1].split()[1]

        result = dict()
        result["PolicyRisk"] = float(PolicyRisk)
        result["ATT"] = float(ATT)
        return result

    def createCommand(self):
        submitParams = dict()
        for label, line in zip(self.paramLabels, self.paramLines):
            key = label.text().lower()
            val = line.text()
            submitParams[key] = val
        temp_dataset = submitParams.pop("dataset")
        options = list()
        for key, val in submitParams.items():
            # skip the parameter if the value  is empty
            if val.strip() == "":
                continue
            key = key.lower()
            option = "--{key} {val}".format(key=key, val=val)
            options.append(option)

        # True if CFRNET, DRNET, SITE, CEVAE, BART, Causual Forest
        if True:
            baseCmd = "python main.py" + " " + self.modelAlias
        # else:
        #     pass
        self.command = baseCmd + " " + " ".join(options) + " --dataset " + temp_dataset
        submitParams["dataset"] = temp_dataset
        self.experiments = submitParams["experiments"]
        self.dataset = submitParams["dataset"]
        print("*****************************************COMMAND RECEIVED*******************************************")
        print(self.command)
        print("********************************************START RUNNING********************************************")

    @pyqtSlot(str)
    def append_text(self, text):
        self.textedit.moveCursor(QTextCursor.End)
        self.textedit.insertPlainText(text)

    def start_process(self):
        self.p = QProcess()
        self.p.readyReadStandardOutput.connect(self.handle_stdout)
        # self.p.readyReadStandardError.connect(self.handle_stderr)
        # self.p.stateChanged.connect(self.handle_state)
        self.p.finished.connect(self.changeBtnStatus)  # Clean up once complete.
        self.p.start(self.command)

    @pyqtSlot()
    def start_thread(self):
        self.objThread = QThread()
        self.obj = threadModel.backgroundApp(self.command, self.dataset, self.modelName, self.experiments)
        self.obj.moveToThread(self.objThread)
        self.obj.finished.connect(self.objThread.quit)
        self.objThread.started.connect(self.obj.run)
        self.objThread.finished.connect(self.changeBtnStatus)
        self.objThread.start()

    def changeBtnStatus(self):
        isUpdated = self.updateButton.isEnabled()
        isRunned = self.runButton.isEnabled()
        if (not isUpdated) and isRunned:
            self.runButton.setEnabled(False)
            self.updateButton.setEnabled(False)

        elif (not isUpdated) and (not isRunned):
            self.runButton.setEnabled(False)
            self.updateButton.setEnabled(True)

        elif isUpdated and not isRunned:
            self.runButton.setEnabled(True)
            self.updateButton.setEnabled(False)

        else:
            print("entered unknown condition")

    def showDetail(self):
        if self.detailsButton.isChecked():
            self.detailsButton.setArrowType(Qt.UpArrow)
            self.detailsButton.setText("Less...")
            self.OptionalParamsBox.show()

        else:
            self.detailsButton.setArrowType(Qt.DownArrow)
            self.detailsButton.setText("More...")
            self.OptionalParamsBox.hide()

    def file_open1(self):
        name, _ = QFileDialog.getOpenFileName(self, 'Open File', options=QFileDialog.DontUseNativeDialog)
        if name == "":
            return 0
        try:
            with open(name, 'r') as file:
                text = file.read()
                # Filter for empty line and Windows carriage return
                text = text.replace("\r\n", "\n")
                text = text.strip()
            self.file_validation(text)

        except:
            msg = QMessageBox(self)
            msg.setIcon(QMessageBox.Critical)
            msg.setText("Error")
            msg.setInformativeText("Please check if the file has the correct format 'key=value'")
            msg.setWindowTitle("Error")
            msg.exec_()

    def file_open2(self):
        name, _ = QFileDialog.getOpenFileName(self, 'Open File', options=QFileDialog.DontUseNativeDialog)
        if name == "":
            return 0
        self.tab2.setConfigLocation(name)

    def file_validation(self, text):
        self.newParamDict = self.delimitParamsDict(text)
        newParamSet = set([param.lower() for param in self.newParamDict.keys()])

        requiredParamSet = set([param.lower() for param in self.args.required])
        optionalParamSet = set([param.lower() for param in self.args.optional])
        unionParamSet = set.union(requiredParamSet, optionalParamSet)
        if newParamSet.issubset(unionParamSet):
            if not requiredParamSet.issubset(newParamSet):
                msg = QMessageBox(self)
                msg.setIcon(QMessageBox.Warning)
                msg.setText("Warning")
                msg.setInformativeText('This file has some missing required parameters')
                msg.setWindowTitle("Warning")
                msg.exec_()
            for label, line in zip(self.paramLabels, self.paramLines):
                key = label.text().lower()
                if key in self.newParamDict.keys():
                    val = self.newParamDict[key]
                    line.setText(val)

        else:
            setDifference = str(newParamSet.difference(unionParamSet))
            msg = QMessageBox(self)
            msg.setIcon(QMessageBox.Critical)
            msg.setText("Error")
            msg.setInformativeText('This file cannot be loaded because it has undefined parameters ' + setDifference)
            msg.setWindowTitle("Error")
            msg.exec_()

    def delimitParamsDict(self, text):
        result = dict()
        delimParamsList = [option.split("=") for option in text.split("\n")]
        for key, val in delimParamsList:
            key = key.lower()
            result[key] = val
        return result

    def close_application(self):
        choice = QMessageBox.question(self, 'Message',
                                      "Are you sure to quit?", QMessageBox.Yes |
                                      QMessageBox.No, QMessageBox.No)

        if choice == QMessageBox.Yes:
            print('quit application')
            sys.exit()
        else:
            pass

    def createResultTabs(self):
        widget = QTabWidget()

        self.tab1 = comparisonTab()
        self.tab2 = hyperparamsTab()

        widget.addTab(self.tab1, "Model Comparisons")
        widget.addTab(self.tab2, "Hyperparameter Search")
        return widget

    def setCenter(self):
        qr = self.frameGeometry()
        cp = QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())

    def handle_stdout(self):
        data = self.p.readAllStandardOutput()
        stdout = bytes(data).decode("utf8")
        print(stdout)


if __name__ == '__main__':
    queue = Queue()
    sys.stdout = WriteStream(queue)

    qapp = QApplication(sys.argv)
    app = MyApp()
    app.show()

    thread = QThread()
    my_receiver = MyReceiver(queue)
    my_receiver.mysignal.connect(app.append_text)
    my_receiver.moveToThread(thread)
    thread.started.connect(my_receiver.run)
    thread.start()
    sys.exit(qapp.exec_())
