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
# Load the data
import argparse
from sklearn.datasets import load_svmlight_file

defaultPath = "."
CLI=argparse.ArgumentParser()
CLI.add_argument(
   "--data_path",
   type=str,
   default=defaultPath
)

args = CLI.parse_args()

X,y = load_svmlight_file(args.data_path + "/data/epsilon_normalized")

# Make the train-test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Convert to dense
import numpy as np
X_train = np.array(X_train.todense())
X_test  = np.array(X_test.todense())

# Write to binary numpy files
np.save(args.data_path + "/data/epsilon.X_train", X_train)
np.save(args.data_path + "/data/epsilon.X_test",  X_test)
np.save(args.data_path + "/data/epsilon.y_train", y_train)
np.save(args.data_path + "/data/epsilon.y_test",  y_test)
