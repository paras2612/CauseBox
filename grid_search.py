import tensorflow as tf
from tensorboard.plugins.hparams import api as hp
import datetime
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
import numpy as np



# Load the TensorBoard notebook extension
%load_ext tensorboard

p_alpha = [-6,-5,-4,-3,-2,-1,0,1,2,3,4,5,6]
p_lambda = [-6,-5,-4,-3,-2,-1,0,1,2,3,4,5,6]
param_grid = dict(p_alpha=p_alpha, p_lambda=p_lambda)
grid = GridSearchCV(estimator=cfr, param_grid=param_grid, n_jobs=-1, cv=3)
X = np.load("D:\PycharmProjects\pythonProject\CFRNet\Dataset\IHDP1\ihdp_npci_1-100.train.npz")
Y = X['yf']
grid_result = grid.fit(X, Y)
# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
