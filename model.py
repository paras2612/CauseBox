from PyQt5.QtCore import QObject, pyqtSlot
from CFRNet import cfr_net_main as cfr
import datetime
import os
import traceback
import sys
from CFRNet.evaluate import *

class backendApp(QObject):
    def __init__(self, command, parent=None):
        QObject.__init__(self, parent)
        self.command = command

    @pyqtSlot()
    def run(self):
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S-%f")
        #outdir = os.getcwd() + '\\CFRNet\\Results\\' + argv[-1].lower().strip() + '\\results_' + timestamp + '\\'
        outdir = os.getcwd() + '\\CFRNet\\Results\\ihdp\\results_' + timestamp + '\\'
        os.makedirs(outdir)
        try:
            cfr.run(outdir,self.command)
        except Exception as e:
            with open(outdir + 'error.txt', 'w') as errfile:
                errfile.write(''.join(traceback.format_exception(*sys.exc_info())))
            print("Some problem occured")

        config_file = outdir + 'config.txt'
        overwrite = False
        filters = None
        evaluate(config_file, overwrite, filters=filters)
