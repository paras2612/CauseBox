from PyQt5.QtCore import QObject, QThread, pyqtSlot, pyqtSignal
import datetime
import traceback
import os
import sys
from CFRNet import cfr_net_main as cfr_model
from CFRNet.evaluate import evaluate as cfr_evaluate
from DRN import dr_cfr as dr_cfr_model
from DRN.evaluate import evaluate as dr_cfr_evaluate
# from CEVAE import cevae as cevae_model
# from CEVAE.evaluation import Evaluator as cevae_evaluate
from SITE import site_net as site_model
from SITE.evaluation import evaluate as site_evaluate

class backgroundApp(QThread):
    finished = pyqtSignal()

    def __init__(self, command, dataset, modelName, experiments):
        super(backgroundApp, self).__init__()
        self.command = command
        self.dataset = dataset
        self.modelName = modelName
        self.experiments = experiments

    @pyqtSlot()
    def run(self):
        if self.modelName == "Counterfactual Regression Network (CFRNet)":
            CFRNet(self.command, self.dataset, self.experiments)
        elif self.modelName == "Causal Effect Inference with Deep Latent-Variable Models (CEVAE)":
            CEVAE(self.command, self.dataset, self.experiments)
        # elif self.modelName == "Bayesian Additive Regression Trees (BART)":
        #     BART(self.command, self.dataset, self.experiments)
        # elif self.modelName == "Causal Forests":
        #     CausalForests(self.command, self.dataset, self.experiments)
        # elif self.modelName == "Perfect Match":
        #     PerfectMatch(self.command, self.dataset, self.experiments)
        elif self.modelName == "Learning Disentangled Representations for counterfactual regression (DRNet)":
            DRNet(self.command, self.dataset, self.experiments)
        elif self.modelName == "Local similarity preserved individual treatment effect (SITE)":
            SITE(self.command, self.dataset, self.experiments)
        else:
            print("Combobox Error occured")
        self.finished.emit()

class CFRNet:
    def __init__(self, command, dataset, experiments):
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S-%f")
        config_dir = os.getcwd() + '\\CFRNet\\Results\\'+dataset.lower().strip()+str(experiments)+'\\results_' + timestamp + '\\'
        print("Output Directory: ", config_dir)
        os.makedirs(config_dir)
        try:
            cfr_model.run(config_dir,command)
        except Exception as e:
            with open(config_dir + 'error.txt', 'w') as errfile:
                errfile.write(''.join(traceback.format_exception(*sys.exc_info())))
            print("Some problem occured")

        config_file = config_dir + 'config.txt'
        overwrite = False
        filters = None
        cfr_evaluate(config_file, overwrite, filters=filters)

class DRNet:
    def __init__(self, command, dataset, experiments):
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S-%f")
        outdir = os.getcwd() + '\\DRN\\Results\\'+dataset.lower().strip()+str(experiments)+'\\results_' + timestamp + '\\'
        print("Output Directory: ", outdir)
        os.makedirs(outdir)
        try:
            dr_cfr_model.run(outdir,command)
        except Exception as e:
            with open(outdir + 'error.txt', 'w') as errfile:
                errfile.write(''.join(traceback.format_exception(*sys.exc_info())))
            print("Some problem occured")

        config_file = outdir + 'config.txt'
        overwrite = False
        filters = None
        dr_cfr_evaluate(config_file, overwrite, filters=filters)

class CEVAE:
    def __init__(self, command, dataset, experiments):
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S-%f")
        outdir = os.getcwd() + '\\CEVAE\\Results\\'+dataset.lower().strip()+str(experiments)+'\\results_' + timestamp + '\\'
        print("Output Directory: ", outdir)
        os.makedirs(outdir)
        try:
            cevae_model.run(outdir,command)
        except Exception as e:
            with open(outdir + 'error.txt', 'w') as errfile:
                errfile.write(''.join(traceback.format_exception(*sys.exc_info())))
            print("Some problem occured")

        config_file = outdir + 'config.txt'
        overwrite = False
        filters = None
        cevae_evaluate(config_file, overwrite, filters=filters)

class SITE:
    def __init__(self, command, dataset, experiments):
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S-%f")
        outdir = os.getcwd() + '\\SITE\\Results\\'+dataset.lower().strip()+str(experiments)+'\\results_' + timestamp + '\\'
        print("Output Directory: ", outdir)
        os.makedirs(outdir)
        try:
            site_model.run(outdir,command)
        except Exception as e:
            with open(outdir + 'error.txt', 'w') as errfile:
                errfile.write(''.join(traceback.format_exception(*sys.exc_info())))
            print("Some problem occured")

        config_file = outdir + 'config.txt'
        overwrite = False
        filters = None
        site_evaluate(config_file, overwrite, filters=filters)