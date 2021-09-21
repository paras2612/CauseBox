import traceback
from sys import argv
from CFRNet.evaluate import *
from SITE import evaluate as ev
from DRN import evaluate as dev
import datetime
from CEVAE.main import main as cevae_main
import subprocess

import logging
import os
from PyQt5.QtCore import QObject, pyqtSlot

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # FATAL
logging.getLogger('tensorflow').setLevel(logging.FATAL)


def main(_):
    if (argv[1].lower() == "cfrnet"):
        import CFRNet.cfr_net_main as cfr
        """ Main entry point """
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S-%f")
        outdir = os.getcwd() + '\\CFRNet\\Results\\' + argv[-1].lower().strip() + '\\results_' + timestamp + '\\'
        os.makedirs(outdir)
        try:
            cfr.run(outdir)
        except Exception as e:
            with open(outdir + 'error.txt', 'w') as errfile:
                errfile.write(''.join(traceback.format_exception(*sys.exc_info())))
            raise
        config_file = outdir + 'config.txt'
        overwrite = False
        filters = None
        evaluate(config_file, overwrite, filters=filters)
    elif argv[1].lower() == "cevae":
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S-%f")
        outdir = os.getcwd() + '\\CEVAE\\Results\\results_' + timestamp + '\\'
        os.makedirs(outdir)
        try:
            cevae_main()
        except Exception as e:
            with open(outdir + 'error.txt', 'w') as errfile:
                errfile.write(''.join(traceback.format_exception(*sys.exc_info())))
            raise

    elif argv[1].lower() == "site":
        import SITE.site_net_train as site
        """ Main entry point """
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S-%f")
        outdir = os.getcwd() + '\\SITE\\Results\\' + argv[-1].lower().strip() + '\\results_' + timestamp + '\\'
        os.makedirs(outdir)
        try:
            site.run(outdir)
        except Exception as e:
            with open(outdir + 'error.txt', 'w') as errfile:
                errfile.write(''.join(traceback.format_exception(*sys.exc_info())))
            raise
        config_file = outdir + 'config.txt'
        # print(config_file)

        print("Training done evaluation started")
        overwrite = False
        filters = None
        ev.evaluate(config_file, overwrite, filters=filters)

    elif argv[1].lower() == "bart":
        process = subprocess.call(
            ["Rscript", sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6], sys.argv[7]])
        if (process == 0):
            print("Check the results in csv file")
        else:
            print("Some error occured. Try Again")

    elif argv[1].lower() == "cforest":
        process = subprocess.call(
            ["Rscript", sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6], sys.argv[7]])
        if (process == 0):
            print("Check the results in csv file")
        else:
            print("Some error occured. Try Again")
    elif argv[1].lower() == "drnet":
        import DRN.dr_cfr_main as drn
        """ Main entry point """
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S-%f")
        outdir = os.getcwd() + '\\DRN\\Results\\' + argv[-1].lower().strip() + '\\results_' + timestamp + '\\'
        os.makedirs(outdir)
        # print(outdir)
        try:
            drn.run(outdir)
        except Exception as e:
            with open(outdir + 'error.txt', 'w') as errfile:
                errfile.write(''.join(traceback.format_exception(*sys.exc_info())))
            raise
        config_file = outdir + 'config.txt'
        # print(config_file)

        # print("Training done evaluation started")
        overwrite = False
        filters = None
        dev.evaluate(config_file, overwrite, filters=filters)
    elif argv[1].lower()=="--pm=pm":
        import PM.main as pm
        from PM.parameters import clip_percentage, parse_parameters
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S-%f")
        outdir = os.getcwd() + '\\PM\\Results\\' + argv[-1].lower().strip() + '\\results_' + timestamp + '\\'
        os.makedirs(outdir)
        # print(outdir)
        try:
            app = pm.MainApplication(parse_parameters())
            app.run()
        except Exception as e:
            with open(outdir + 'error.txt', 'w') as errfile:
                errfile.write(''.join(traceback.format_exception(*sys.exc_info())))
            raise


if __name__ == "__main__":
    main(argv)