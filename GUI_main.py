import sys
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
# import model
import args
from queue import Queue
from GUI_tab import Tab1, Tab2
import json
import os
import datetime
# import sip

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
        self.modelName = "Counterfactual Regression Network (CFRNet)"
        self.MainWindowGUI()

    def MainWindowGUI(self):
        exitApp = QAction('&Close Window', self)
        exitApp.setShortcut('Ctrl+Q')
        exitApp.setStatusTip('leave the app')
        exitApp.triggered.connect(self.close_application)

        openFile = QAction('&Open File', self)
        openFile.setShortcut('Ctrl+O')
        openFile.setStatusTip('Open File')
        openFile.triggered.connect(self.file_open)
        fileMenu = self.menuBar().addMenu('&File')
        fileMenu.addAction(openFile)
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
        self.setLayout(self.mainLayout)

        self.mainWidget = QWidget()
        self.mainWidget.setLayout(self.mainLayout)
        self.setCentralWidget(self.mainWidget)

        self.setWindowIcon(self.style().standardIcon(getattr(QStyle, 'SP_FileDialogListView')))
        self.setWindowTitle('CIKM Demo Tutorial')
        self.setGeometry(2000, 2000, 2000, 2000)
        self.setCenter()

    def createButtonBox(self):
        widget= QWidget()
        layout = QHBoxLayout()
        self.saveButton = QToolButton()
        self.saveButton.setText("Save The Result To Database √")
        self.saveButton.setEnabled(False)
        self.runButton = QToolButton()
        self.runButton.setText("Run The Model With Parameter √")
        self.saveButton.clicked.connect(self.saveResult)
        self.runButton.clicked.connect(self.runModel)

        layout.addWidget(self.runButton)
        layout.addWidget(self.saveButton)
        widget.setLayout(layout)
        return widget

    def createParamsBox(self):
        widget = QWidget(self)
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
        modelComboBox.addItem("Counterfactual Regression Network (CRFNet)")
        modelComboBox.addItem("Causal Effect Inference with Deep Latent-Variable Models (CEVAE)")
        modelComboBox.addItem("Bayesian Additive Regression Trees (BART)")
        modelComboBox.addItem("Causal Forests")
        modelComboBox.addItem("Perfect Match")
        modelComboBox.addItem("Learning Disentangled Representations for counterfactual regression (DRNet)")
        modelComboBox.addItem("Local similarity preserved individual treatment effect (SITE)")
        modelComboBox.activated[str].connect(self.modelChoice)

        dataComboBox = QComboBox()
        dataComboBox.addItem("Jobs")
        dataComboBox.addItem("IHDP")
        dataComboBox.activated[str].connect(self.dataChoice)

        self.SettingLayout = QVBoxLayout()
        self.SettingLayout.addWidget(modelComboBox)
        self.SettingLayout.addWidget(dataComboBox)

        widget.setLayout(self.SettingLayout)
        return widget

    def modelChoice(self, text):
        self.modelName = text
        if self.modelName == "Counterfactual Regression Network (CRFNet)":
            self.args = args.CFRNet()
        elif self.modelName == "Causal Effect Inference with Deep Latent-Variable Models (CEVAE)":
            self.args = args.CEVAE()
        elif self.modelName == "Bayesian Additive Regression Trees (BART)":
            self.args = args.BART()
        elif self.modelName == "Causal Forests":
            self.args = args.CausalForests()
        elif self.modelName == "Perfect Match":
            self.args = args.PerfectMatch()
        elif self.modelName == "Learning Disentangled Representations for counterfactual regression (DRNet)":
            self.args = args.DRNet()
        elif self.modelName == "Local similarity preserved individual treatment effect (SITE)":
            self.args = args.SITE()
        else:
            print("Combobox Error occured")

        self.updateWidget()

    def updateWidget(self):
        # self.updateSettingsWidget()
        self.updateParamsWidget()
        # self.updateConsoleWidget()

    def updateSettingsWidget(self):
        if hasattr(self, 'SettingsBox'):
            self.mainLayout.removeWidget(self.SettingsBox)
            self.SettingsBox = self.createSettingsBox()
            self.mainLayout.addWidget(self.SettingsBox,0,0)
            self.mainLayout.update()

    def updateParamsWidget(self):
        if hasattr(self, 'ParamsBox'):
            self.mainLayout.removeWidget(self.ParamsBox)
            # sip.delete(self.ParamsBox)
            # self.ParamsBox = None
            self.ParamsBox = self.createParamsBox()

            self.mainLayout.addWidget(self.ParamsBox, 1, 0)
            self.mainWidget.setLayout(self.mainLayout)
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

    def saveResult(self):
        self.reset()
        self.changeBtnStatus()

    def updateStatus(self):
        self.changeBtnStatus()
        outdir = self.newParamDict["outdir"]
        timestamp =datetime.datetime.now().strftime("%Y%m%d_%H%M%S-%f")

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
        self.command = "python main.py "
        options = list()
        for key, val in submitParams.items():
            # skip the parameter if the value  is empty
            if val.strip() == "":
                continue
            key = key.lower()
            option = "--{key} {val}".format(key=key, val=val)
            options.append(option)

        self.command = self.command + " ".join(options) + " --dataset %s" % (self.dataset)
        self.experiments = submitParams["experiments"]
        print(self.command)
    @pyqtSlot(str)
    def append_text(self, text):
        self.textedit.moveCursor(QTextCursor.End)
        self.textedit.insertPlainText(text)

    @pyqtSlot()
    def start_thread(self):
        self.thread = QThread()
        self.backendApp = model.backendApp(self.command, self.dataset, self.modelName, self.experiments)
        self.backendApp.moveToThread(self.thread)
        self.thread.setTerminationEnabled(True)
        self.thread.started.connect(self.backendApp.run)
        self.thread.start()

    def changeBtnStatus(self):
        if self.runButton.isEnabled():
            self.runButton.setEnabled(False)
            self.saveButton.setEnabled(True)

        else:
            self.runButton.setEnabled(True)
            self.saveButton.setEnabled(False)

    def showDetail(self):
        if self.detailsButton.isChecked():
            self.detailsButton.setArrowType(Qt.UpArrow)
            self.detailsButton.setText("Less...")
            self.OptionalParamsBox.show()

        else:
            self.detailsButton.setArrowType(Qt.DownArrow)
            self.detailsButton.setText("More...")
            self.OptionalParamsBox.hide()

    def file_open(self):
        name, _ = QFileDialog.getOpenFileName(self, 'Open File', options=QFileDialog.DontUseNativeDialog)
        if name == "":
            return 0
        file = open(name, 'r')
        try:
            with file:
                text = file.read()
                text = text.replace("\r\n", "\n")
                text = text.strip()
            self.file_validation(text)

        except:
            msg = QMessageBox(self)
            msg.setIcon(QMessageBox.Critical)
            msg.setText("Error")
            msg.setInformativeText("Please check if you selected has the correct format")
            msg.setWindowTitle("Error")
            msg.exec_()

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
            msg = QMessageBox(self)
            msg.setIcon(QMessageBox.Critical)
            msg.setText("Error")
            msg.setInformativeText('This file cannot be loaded because it has undefined parameters')
            msg.setWindowTitle("Error")
            msg.exec_()

    def delimitParamsDict(self, text):
        result = dict()
        delimParamsList = [option.split() for option in text.split("\n")]
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

        self.rankTab = Tab1()
        # self.expTab = Tab2()

        widget.addTab(self.rankTab, "Comparison")
        # widget.addTab(self.expTab, "Hyperparameter Sesarch")
        return widget

    def setCenter(self):
        qr = self.frameGeometry()
        cp = QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())
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