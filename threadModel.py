from PyQt5.QtCore import QObject, pyqtSlot, pyqtSignal
import datetime
import os
import traceback
import sys
from CFRNet import cfr_net_main as cfr
from CFRNet.evaluate import evaluate

class backgroundApp(QObject):
    finished = pyqtSignal()
    def __init__(self, command,  outdir, parent=None):
        QObject.__init__(self, parent)
        self.command = command
        self.outdir = outdir

    @pyqtSlot()
    def run(self):

        # timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S-%f")
        #outdir = os.getcwd() + '\\CFRNet\\Results\\' + argv[-1].lower().strip() + '\\results_' + timestamp + '\\'
        # outdir = os.getcwd() + '\\CFRNet\\Results\\ihdp\\results_' + timestamp + '\\'
        # os.makedirs(outdir)
        try:
            cfr.run(self.outdir, self.command)
        except Exception as e:
            # with open('./error.txt', 'w') as errfile:
            #     errfile.write(''.join(traceback.format_exception(*sys.exc_info())))
            print("Some problem occured")

        config_file = self.outdir + '/config.txt'
        overwrite = False
        filters = None
        # evaluate(config_file, overwrite, filters=filters)
        self.finished.emit()
