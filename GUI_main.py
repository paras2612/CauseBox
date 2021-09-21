import sys
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
import model
import args
from queue import Queue

command = ""

class WriteStream(object):
    def __init__(self, queue):
        self.queue = queue

    def write(self, text):
        self.queue.put(text)


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
        self.dataset = "IHDP"
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

        # self.progressBar = QProgressBar()
        # self.progressBar.setAlignment(Qt.AlignVCenter)

        buttonBox = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttonBox.button(QDialogButtonBox.Ok).setText("Submit")
        buttonBox.button(QDialogButtonBox.Cancel).setText("Reset")
        buttonBox.accepted.connect(self.accept)
        buttonBox.rejected.connect(self.reject)

        self.detailsButton = QToolButton()
        self.detailsButton.setText("Less...")
        self.detailsButton.setCheckable(True)
        self.detailsButton.setChecked(True)
        self.detailsButton.setArrowType(Qt.UpArrow)
        self.detailsButton.setAutoRaise(True)
        self.detailsButton.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.detailsButton.clicked.connect(self.showDetail)

        self.createRequiredParamsBox()
        self.createOptionalParamsBox()
        self.createConsoleBox()
        self.createSettingsBox()

        self.mainLayout = QGridLayout()
        self.mainLayout.addWidget(self.SettingsBox, 0, 0)
        self.mainLayout.addWidget(self.RequiredParamasBox, 1, 0)
        self.mainLayout.addWidget(self.detailsButton, 2, 0)
        self.mainLayout.addWidget(self.OptionalParamsBox, 3, 0)
        self.mainLayout.addWidget(self.ConsoleBox, 4, 0)
        # self.mainLayout.addWidget(self.progressBar, 5, 0)
        self.mainLayout.addWidget(buttonBox, 5, 0)
        self.setLayout(self.mainLayout)

        mainWidget = QWidget()
        mainWidget.setLayout(self.mainLayout)
        self.setCentralWidget(mainWidget)

        self.setWindowIcon(self.style().standardIcon(getattr(QStyle, 'SP_FileDialogListView')))
        self.setWindowTitle('CIKM Demo Tutorial')
        self.setGeometry(1000, 1000, 1000, 1000)
        self.setCenter()

    def createRequiredParamsBox(self):
        self.RequiredParamasBox = QGroupBox("Required Parameters")
        layout = QGridLayout()

        self.labels = list()
        self.lines = list()
        i = j = 0
        for key in self.args.required:
            i += 1
            label = QLabel(key)
            line = QLineEdit()
            layout.addWidget(label, i, j)
            layout.addWidget(line, i, j + 1)
            self.labels.append(label)
            self.lines.append(line)

            #Move to next column if the rows are more than 4 lines
            if i > 4:
                i = 0
                j += 2

        self.RequiredParamasBox.setLayout(layout)

    def createOptionalParamsBox(self):
        self.OptionalParamsBox = QGroupBox("Optional Parameters")
        layout = QGridLayout()

        i = j = 0
        for key in self.args.optional:
            i += 1
            label = QLabel(key)
            line = QLineEdit()
            layout.addWidget(label, i, j)
            layout.addWidget(line, i, j + 1)
            self.labels.append(label)
            self.lines.append(line)

            #Move to next column if the rows are more than 4 lines
            if i >= 4:
                i = 0
                j += 2

        self.OptionalParamsBox.setLayout(layout)

    def createConsoleBox(self):
        self.ConsoleBox = QGroupBox("Console")
        self.textedit = QTextEdit()

        layout = QVBoxLayout()
        layout.addWidget(self.textedit)

        self.ConsoleBox.setLayout(layout)

    def createSettingsBox(self):
        self.SettingsBox = QGroupBox("Settings")

        modelComboBox = QComboBox()
        modelComboBox.addItem("Counterfactual Regression Network (CFRNet)")
        modelComboBox.addItem("Causal Effect Inference with Deep Latent-Variable Models (CEVAE)")
        modelComboBox.addItem("Bayesian Additive Regression Trees (BART)")
        modelComboBox.addItem("Causal Forests")
        modelComboBox.addItem("Perfect Match")
        modelComboBox.addItem("Learning Disentangled Representations for counterfactual regression (DRNet)")
        modelComboBox.addItem("Local similarity preserved individual treatment effect (SITE)")
        modelComboBox.activated[str].connect(self.modelChoice)

        dataComboBox = QComboBox()
        dataComboBox.addItem("IHDP")
        dataComboBox.addItem("Jobs")
        dataComboBox.activated[str].connect(self.dataChoice)

        layout = QVBoxLayout()
        layout.addWidget(modelComboBox)
        layout.addWidget(dataComboBox)

        self.SettingsBox.setLayout(layout)
        self.reset()

    def modelChoice(self, text):
        self.mainLayout.removeWidget(self.RequiredParamasBox)
        self.mainLayout.removeWidget(self.OptionalParamsBox)
        if text == "Counterfactual Regression Network (CRFNet)":
            self.args = args.CFRNet()
        elif text == "Causal Effect Inference with Deep Latent-Variable Models (CEVAE)":
            self.args = args.CEVAE()
        elif text == "Bayesian Additive Regression Trees (BART)":
            self.args = args.BART()
        elif text == "Causal Forests":
            self.args = args.CausalForests()
        elif text == "Perfect Match":
            self.args = args.PerfectMatch()
        elif text == "Learning Disentangled Representations for counterfactual regression (DRNet)":
            self.args = args.DRNet()
        elif text == "Local similarity preserved individual treatment effect (SITE)":
            self.args = args.SITE()
        else:
            print("not implemented")
        self.createRequiredParamsBox()
        self.createOptionalParamsBox()

        self.mainLayout.addWidget(self.RequiredParamasBox, 1, 0)
        self.mainLayout.addWidget(self.OptionalParamsBox, 3, 0)

        self.RequiredParamasBox.update()
        self.OptionalParamsBox.update()

    def dataChoice(self, text):
        if text == "IHDP":
            self.dataset = "IHDP"
        elif text == "Jobs":
            self.dataset = "Jobs"

    def accept(self):
        self.createCommand()
        self.start_thread()

    def reject(self):
        self.reset()

    def createCommand(self):
        submitParams = dict()
        for label, line in zip(self.labels, self.lines):
            key = label.text()
            val = line.text()
            submitParams[key] = val
        global command
        command = "python main.py "
        options = list()
        for key, val in submitParams.items():
            #skip the parameter if the value  is empty
            if val.strip() == "":
                continue
            option = "--{key} {val}".format(key=key, val=val)
            options.append(option)

        command = command + " ".join(options)
        print("*" * 10, "Testing Command Line", "*" * 10)
        print(command)
        print("*" * 40)

    @pyqtSlot(str)
    def append_text(self, text):
        self.textedit.moveCursor(QTextCursor.End)
        self.textedit.insertPlainText(text)

    @pyqtSlot()
    def start_thread(self):
        self.thread = QThread()
        self.backendApp = model.backendApp(command)
        self.backendApp.moveToThread(self.thread)
        self.thread.started.connect(self.backendApp.run)
        self.thread.start()

    def reset(self):
        lines = self.lines
        for i, line in enumerate(lines):
            line.setText("")
        # self.textedit.setText("")

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
        file = open(name, 'r')
        with file:
            text = file.read()
            text = text.replace("\r\n", "\n")
            text = text.strip()
        self.file_validation(text)

    def file_validation(self, text):
        loadedParams = dict()
        delimParams = [option.split(" ") for option in text.split("\n")]
        for key, val in delimParams:
            loadedParams[key] = val
        if list(loadedParams.keys()) == self.args.required:
            for label, line in zip(self.labels, self.lines):
                key = label.text()
                val = loadedParams[key]
                line.setText(val)

        else:
            msg = QMessageBox(self)
            msg.setIcon(QMessageBox.Critical)

    def close_application(self):
        choice = QMessageBox.question(self, 'Message',
                                      "Are you sure to quit?", QMessageBox.Yes |
                                      QMessageBox.No, QMessageBox.No)

        if choice == QMessageBox.Yes:
            print('quit application')
            sys.exit()
        else:
            pass

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
