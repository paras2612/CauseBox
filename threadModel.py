from PyQt5.QtCore import QObject, QThread, pyqtSlot, pyqtSignal
import datetime
import traceback
import os
import sys
from CFRNet.evaluate import evaluate as cfr_evaluate
from SITE import evaluate as ev
from DRN import evaluate as dev
import subprocess
import random

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
        # elif self.modelName == "Causal Effect Inference with Deep Latent-Variable Models (CEVAE)":
        #     CEVAE(self.command, self.dataset, self.experiments)
        elif self.modelName == "Bayesian Additive Regression Trees (BART)":
            BART(self.command, self.dataset, self.experiments)
        # elif self.modelName == "Causal Forests":
        #     CausalForests(self.command, self.dataset, self.experiments)
        # elif self.modelName == "Perfect Match":
        #     PerfectMatch(self.command, self.dataset, self.experiments)
        elif self.modelName == "Learning Disentangled Representations for counterfactual regression (DRNet)":
            DRNet(self.command, self.dataset, self.experiments)
        elif self.modelName == "Local similarity preserved individual treatment effect (SITE)":
            SITE(self.command, self.dataset, self.experiments)
        else:
            print("The application cannot support this model right now")
        self.finished.emit()

class CFRNet:
    def __init__(self, command, dataset, experiments):
        from CFRNet import cfr_net_main as cfr_model
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S-%f")
        config_dir = os.getcwd() + '\\CFRNet\\Results\\'+dataset.lower().strip()+str(experiments)+'\\results_' + timestamp + '\\'
        print("Output Directory: ", config_dir)
        os.makedirs(config_dir)
        try:
            cfr_model.run(config_dir,command)
        except Exception as e:
            with open(config_dir + 'error.txt', 'w') as errfile:
                errfile.write(''.join(traceback.format_exception(*sys.exc_info())))

        config_file = config_dir + 'config.txt'
        overwrite = False
        filters = None
        cfr_evaluate(config_file, overwrite, filters=filters)
        #result = "PEHE is  {} and error in ATE is  {}".format(metric1, metric2)
        #print(result)
        #line = "CFRNET, ihdp,1.5593514529612729," + str(metric1)
        #with open(os.getcwd() +"\\tempResult.csv", "w") as f:
        #    f.write(line)

class DRNet:
    def __init__(self, command, dataset, experiments):
        import DRN.dr_cfr_main as drn
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S-%f")
        outdir = os.getcwd() + '\\DRN\\Results\\'+dataset.lower().strip()+str(experiments)+'\\results_' + timestamp + '\\'
        print("Output Directory: ", outdir)
        os.makedirs(outdir)
        try:
            drn.run(outdir, command)
        except Exception as e:
            with open(outdir + 'error.txt', 'w') as errfile:
                errfile.write(''.join(traceback.format_exception(*sys.exc_info())))

        config_file = outdir + 'config.txt'
        overwrite = False
        filters = None
        #ERROR occur: TypeError: only size-1 arrays can be converted to Python scalars
        dev.evaluate(config_file, overwrite, filters=filters)

class CEVAE:
    def __init__(self, command, dataset, experiments):
        from CEVAE.main import main as cevae_main
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S-%f")
        outdir = os.getcwd() + '\\CEVAE\\Results\\'+dataset.lower().strip()+str(experiments)+'\\results_' + timestamp + '\\'
        print("Output Directory: ", outdir)
        os.makedirs(outdir)
        try:
            #How to read parameter args like site and cfrnet?
            cevae_main()
        except Exception as e:
            with open(outdir + 'error.txt', 'w') as errfile:
                errfile.write(''.join(traceback.format_exception(*sys.exc_info())))
            print("Some problem occured")

class SITE:
    def __init__(self, command, dataset, experiments):
        import SITE.site_net_train as site
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S-%f")
        outdir = os.getcwd() + '\\SITE\\Results\\'+dataset.lower().strip()+str(experiments)+'\\results_' + timestamp + '\\'
        # print("Output Directory: ", outdir)
        os.makedirs(outdir)
        try:
            site.run(outdir,command)
        except Exception as e:
            with open(outdir + 'error.txt', 'w') as errfile:
                errfile.write(''.join(traceback.format_exception(*sys.exc_info())))
            print(e)

        config_file = outdir + 'config.txt'
        overwrite = False
        filters = None
        ev.evaluate(config_file, overwrite, filters=filters)

# class PerfectMatch:
#     def __init__(self, command, dataset, experiments):
#         import PM.main as pm
#         from PM.parameters import clip_percentage, parse_parameters
#         timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S-%f")
#         config_dir = os.getcwd() + '\\CFRNet\\Results\\'+dataset.lower().strip()+str(experiments)+'\\results_' + timestamp + '\\'
#         print("Output Directory: ", config_dir)
#         os.makedirs(config_dir)
#         try:
#             app = pm.MainApplication(parse_parameters())
#             app.run()#         except Exception as e:
#             with open(config_dir + 'error.txt', 'w') as errfile:
#                 errfile.write(''.join(traceback.format_exception(*sys.exc_info())))
#
#         config_file = config_dir + 'config.txt'
#         overwrite = False
#         filters = None
#         cfr_evaluate(config_file, overwrite, filters=filters)
class BART:
    def __init__(self, command, dataset, experiments):
        command_list = command.split()
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S-%f")
        config_dir = os.getcwd() + '\\BART\\Results\\'+dataset.lower().strip()+str(experiments)+'\\results_' + timestamp + '\\'
        print("Output Directory: ", config_dir)
        os.makedirs(config_dir)
        # Result = 5?
        print(os.getcwd()+command_list[3])
        print(os.getcwd() + command_list[5])
        print(os.getcwd() + command_list[7])
        print(os.getcwd() + command_list[9])
        print(command_list[11])
        print(command_list[13])
        try:
            process = subprocess.call(["Rscript", (os.getcwd()+command_list[3]).strip(), (os.getcwd()+command_list[5]).strip(), (os.getcwd()+command_list[7]).strip(), (os.getcwd()+command_list[9]).strip(), command_list[11], command_list[13]])
            if (process == 0):
                print("Check the results in csv file")
            else:
                print("Some error occured. Try Again")
                print(process)
        except:
            with open(config_dir + 'error.txt', 'w') as errfile:
                errfile.write(''.join(traceback.format_exception(*sys.exc_info())))
# class CausalForests:
#     def __init__(self, command, dataset, experiments):
#         import PM.main as pm
#         from PM.parameters import clip_percentage, parse_parameters
#         timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S-%f")
#         config_dir = os.getcwd() + '\\CFRNet\\Results\\'+dataset.lower().strip()+str(experiments)+'\\results_' + timestamp + '\\'
#         print("Output Directory: ", config_dir)
#         os.makedirs(config_dir)
#         try:
#         process = subprocess.call(
#             ["Rscript", sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6], sys.argv[7]])#         except Exception as e:
#             with open(config_dir + 'error.txt', 'w') as errfile:
#                 errfile.write(''.join(traceback.format_exception(*sys.exc_info())))
#
#         config_file = config_dir + 'config.txt'
#         overwrite = False
#         filters = None
#         cfr_evaluate(config_file, overwrite, filters=filters)
