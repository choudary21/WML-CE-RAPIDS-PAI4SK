# *****************************************************************
#
# Licensed Materials - Property of IBM
#
# (C) Copyright IBM Corp. 2019. All Rights Reserved.
#
# US Government Users Restricted Rights - Use, duplication or
# disclosure restricted by GSA ADP Schedule Contract with IBM Corp.
#
# *****************************************************************


import numpy as np
from numpy import genfromtxt
import pandas as pd

ytrain = np.load('./data/epsilon.y_train.npy')
ytest = np.load('./data/epsilon.y_test.npy')
Xtrain = np.load('./data/epsilon.X_train.npy')
Xtest = np.load('./data/epsilon.X_test.npy')

Xtrain = np.asarray(Xtrain)
Xtest = np.asarray(Xtest)
ytrain = np.asarray(ytrain)
ytest = np.asarray(ytest)

np.savetxt("./data/epsilon.X_train.csv", Xtrain, delimiter=",")
np.savetxt("./data/epsilon.X_test.csv", Xtest, delimiter=",")
np.savetxt("./data/epsilon.y_train.csv", ytrain, delimiter=",")
np.savetxt("./data/epsilon.y_test.csv", ytest, delimiter=",")
