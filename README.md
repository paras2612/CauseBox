# CauseBox-A-Causal-Inference-Toolbox-for-BenchmarkingTreatment-Effect-Estimators-with-Machine-Learning-Methods
Causal inference is a critical task in various fields such as healthcare,economics, marketing and education. Recently, there have beensignificant advances through the application of machine learningtechniques, especially deep neural networks. Unfortunately, to-datemany of the proposed methods are evaluated on different (data,software/hardware, hyperparameter) setups and consequently it isnearly impossible to compare the efficacy of the available methodsor reproduce results presented in original research manuscripts.In this paper, we propose a causal inference toolbox (CauseBox)that addresses the aforementioned problems. At the time of thewriting, the toolbox includes seven state of the art causal inferencemethods and two benchmark datasets. By providing convenientcommand-line and GUI-based interfaces, theCauseBoxtoolboxhelps researchers fairly compare the state of the art methods intheir chosen application context against benchmark datasets.

# Usage
1) Uncompress datasets for IHDP before you use it as followings:

In Windows, use the command <code>.DatasetScripts/IHDP_uncompress.bat</code>

In Linux, use the command <code>.DatasetScripts/IHDP_uncompress.sh</code>

2) Please download R(version==4.08) on the internet

3) Run the GUI using the command:
<code>python GUI_main.py</code>
